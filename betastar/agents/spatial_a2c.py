from typing import List, Tuple

from gym import spaces
import numpy as np
import torch as T
import torch
import wandb
from betastar.agents import base_agent
from betastar.data import Trajectory, UnrollDataset, collate
from betastar.envs.env import Action, ActionMask, PySC2Env, Value, spawn_env
from betastar.player import Player
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange


class ScreenNet(torch.nn.Module):
    """Model for movement based mini games in sc2.
    This network only takes screen input and only returns spatial outputs.
    Some of the example min games are MoveToBeacon and CollectMineralShards.
    Arguments:
        - in_channel: Number of feature layers in the screen input
        - screen_size: Screen size of the mini game. If 64 is given output
            size will be 64*64
    Note that output size depends on screen_size.
    """

    class ResidualConv(torch.nn.Module):
        def __init__(self, in_channel, **kwargs):
            super().__init__()
            assert kwargs["out_channels"] == in_channel, (
                "input channel must" "be the same as out" " channels"
            )
            self.block = torch.nn.Sequential(
                torch.nn.Conv2d(**kwargs),
                torch.nn.InstanceNorm2d(in_channel),
                torch.nn.ReLU(),
                torch.nn.Conv2d(**kwargs),
                torch.nn.InstanceNorm2d(in_channel),
                torch.nn.ReLU(),
            )

        def forward(self, x):
            res_x = self.block(x)
            return res_x + x

    def __init__(self, in_channel, screen_size):
        super().__init__()
        res_kwargs = {
            "in_channels": 32,
            "out_channels": 32,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
        }
        self.convnet = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, 32, 3, 1, padding=1),
            self.ResidualConv(32, **res_kwargs),
            self.ResidualConv(32, **res_kwargs),
            self.ResidualConv(32, **res_kwargs),
            self.ResidualConv(32, **res_kwargs),
        )

        self.policy = torch.nn.Sequential(
            torch.nn.Linear(32 * screen_size * screen_size, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2 * screen_size),
        )

        self.value = torch.nn.Sequential(
            torch.nn.Linear(32 * screen_size * screen_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

        self.screen_size = screen_size
        gain = torch.nn.init.calculate_gain("relu")

        def param_init(module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight, gain)
                torch.nn.init.zeros_(module.bias)
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.dirac_(module.weight)
                torch.nn.init.zeros_(module.bias)  # type: ignore

        self.apply(param_init)

    def forward(self, state):
        encode = self.convnet(state / 255)

        value = self.value(encode.reshape(encode.shape[0], -1).detach())

        logits = self.policy(encode).squeeze(1)

        return logits, value


class MultiCategorical:
    def __init__(self, *logits):
        self.dists = [Categorical(logits=logit) for logit in logits]

    def sample(self):
        return [dist.sample() for dist in self.dists]

    def log_prob(self, *acts):
        return sum(dist.log_prob(act) for act, dist in zip(acts, self.dists))

    def entropy(self):
        return sum(dist.entropy() for dist in self.dists)

    @property
    def greedy_action(self):
        return [torch.argmax(dist.logits, dim=-1) for dist in self.dists]


class SpatialA2C(base_agent.BaseAgent):
    """
    Plays MoveToBeaconSimple (a spatial version of MoveToBeacon with action space (spatial_dim, spatial_dim))
    """

    def dataloader(self, trajectories: List[Trajectory]) -> DataLoader:
        return DataLoader(
            UnrollDataset(trajectories),
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate,
        )

    def run(self):
        device = T.device("cuda" if T.cuda.is_available() else "cpu")  # type: ignore
        cpu = T.device("cpu")  # type: ignore

        env = spawn_env(
            self.config.environment,
            self.config.game_speed,
            spatial_dim=self.config.screen_size,
            rank=-1,
        )
        # model = ActorCritic(env=env).to(device)
        model = ScreenNet(
            env.feature_original_shapes[0][0], self.config.screen_size
        ).to(device)
        wandb.watch(model, log="all")

        max_reward = 0

        if self.config.use_ppo:
            previous_model = ScreenNet(
                env.feature_original_shapes[0][0], self.config.screen_size
            ).to(device)
            previous_model.load_state_dict(model.state_dict())  # type: ignore

        env.close()

        opt = Adam(
            model.parameters(), lr=self.config.learning_rate, betas=(0.92, 0.999)
        )

        total_cycles = (
            self.config.total_steps
            // self.config.unroll_length
            * self.config.num_workers
        )

        cycles = 0

        # set up environment
        screen, minimap, non_spatial, _reward, _done, action_mask = self.env.reset()
        screen = screen.to(device)

        eps_rewards = np.zeros((self.config.num_workers, 1))
        reward_list = [0]

        step_n = 0
        with tqdm(range(self.config.total_steps), unit="steps", leave=True) as pbar:
            while step_n < self.config.total_steps:
                # if self.config.anneal_lr:
                #     frac = 1.0 - (cycles - 1.0) / total_cycles
                #     lrnow = frac * self.config.learning_rate
                #     opt.param_groups[0]['lr'] = lrnow

                # player.reload_from(model)

                # play
                transitions = []

                for i in tqdm(
                    range(self.config.unroll_length), unit="steps", leave=False
                ):
                    logits, value = model(screen)

                    dist = MultiCategorical(*logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(*action)

                    if self.config.use_ppo:
                        with T.no_grad():
                            old_logits, _old_value = previous_model(screen)  # type: ignore
                            old_log_prob = MultiCategorical(*old_logits).log_prob(
                                *action
                            )
                    else:
                        # we won't use this
                        old_log_prob = log_prob

                    entropy = dist.entropy()
                    action = action[0] * self.config.screen_size + action[1]

                    (
                        screen,
                        _minimap,
                        _non_spatial,
                        reward,
                        done,
                        _action_mask,
                    ) = self.env.step(action.detach().to(cpu))
                    screen = screen.to(device)

                    with torch.no_grad():
                        (
                            _,
                            next_value,
                        ) = model(screen)

                    transitions.append(
                        (
                            reward,
                            done,
                            log_prob,
                            value.squeeze(),
                            next_value.squeeze(),
                            entropy,
                            old_log_prob,
                        )
                    )

                    for j, d in enumerate(done.flatten()):
                        eps_rewards[j] += reward[j].item()
                        if d == 1:
                            reward_list.append(eps_rewards[j].item())
                            eps_rewards[j] = 0

                if self.config.use_ppo:
                    previous_model.load_state_dict(model.state_dict())  # type: ignore

                policy_losses = []
                critic_losses = []
                entropy_losses = []
                losses = []

                # epochs = self.config.update_epochs if self.config.use_ppo else 1
                # for i in range(epochs):
                # learn!
                # all things are whatever x batch_size
                rewards = T.stack([x[0] for x in transitions]).to(device)
                dones = T.stack([x[1] for x in transitions]).to(device)
                log_probs = T.stack([x[2] for x in transitions]).to(device)
                values = T.stack([x[3] for x in transitions]).to(device)
                next_values = T.stack([x[4] for x in transitions]).to(device)
                entropies = T.stack([x[5] for x in transitions]).to(device)
                old_log_probs = T.stack([x[6] for x in transitions]).to(device)

                if self.config.use_gae:
                    advantages = []
                    advantage = 0
                    next_value = (1 - dones[-1].long()) * next_values[-1]

                    for i in reversed(range(len(values))):
                        td_error = (
                            rewards[i]
                            + next_value * self.config.reward_decay
                            - values[i]
                        )
                        advantage = (
                            td_error
                            + advantage
                            * self.config.reward_decay
                            * self.config.trace_decay
                        )
                        next_value = values[i]
                        advantages.insert(0, advantage)

                    advantages = T.stack(advantages).to(device)
                    returns = advantages + values
                else:
                    # vanilla returns with bootstrap
                    R = next_values[-1]
                    rets = []
                    for i in reversed(range(len(values))):
                        R = (
                            R * (1 - dones[i].long()) * self.config.reward_decay
                            + rewards[i]
                        )
                        rets.insert(0, R)
                    returns = T.stack(rets).to(device)
                    advantages = returns - values

                if self.config.normalize_returns:
                    returns = (returns - returns.mean()) / (
                        returns.std() + np.finfo(np.float32).eps.item()
                    )
                if self.config.normalize_advantages:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + np.finfo(np.float32).eps.item()
                    )

                value_loss = advantages.pow(2).mean() * self.config.critic_coeff

                if self.config.use_ppo:
                    surrogate_objective = (log_probs - old_log_probs).exp()
                    clipped_surrogate_objective = surrogate_objective.clamp(
                        min=1.0 - self.config.clip_range,
                        max=1.0 + self.config.clip_range,
                    )
                    policy_loss = T.min(
                        surrogate_objective * advantages.detach(),
                        clipped_surrogate_objective * advantages.detach(),
                    ).mean()
                else:
                    policy_loss = (-log_probs * advantages.detach()).mean()

                entropy_loss = entropies.mean() * self.config.entropy_coeff

                loss = value_loss + policy_loss - entropy_loss
                policy_losses.append(policy_loss.item())
                critic_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                losses.append(loss.item())

                opt.zero_grad()
                loss.backward()
                opt.step()

                metrics = {
                    "episode_reward/last10/mean": np.mean(reward_list[-10:]),
                    "episode_reward/last10/max": np.max(reward_list[-10:]),
                    "episode_reward/last50/mean": np.mean(reward_list[-50:]),
                    "episode_reward/last50/max": np.max(reward_list[-50:]),
                    "loss/real": np.mean(losses),
                    "loss/actor": np.mean(policy_losses),
                    "loss/critic": np.mean(critic_losses),
                    "loss/entropy": np.mean(entropy_losses),
                }

                _max_reward = np.max(reward_list)

                if _max_reward > max_reward:
                    wandb.run.summary["episode_reward/max"] = _max_reward
                    max_reward = _max_reward

                if (
                    cycles > 0 and cycles % self.config.test_interval == 0
                ) or step_n > self.config.total_steps:
                    metrics["video"] = self.last_video(step_n)
                    wandb.log_artifact(
                        self.last_replay(
                            map_name=self.config.environment, step_n=step_n
                        )
                    )

                wandb.log(metrics, step=step_n, commit=True)

                cycles += 1

                _played_steps = self.config.unroll_length * self.config.num_workers
                pbar.update(_played_steps)
                step_n += _played_steps

        wandb.finish()
        self.env.stop()
