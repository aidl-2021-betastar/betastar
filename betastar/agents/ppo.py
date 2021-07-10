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
from torch import nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from tqdm import tqdm


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



class PPO(base_agent.BaseAgent):
    def interpret_action(self, action: List[T.Tensor]) -> T.Tensor:
        raise NotImplementedError()

    def get_model(self, env: PySC2Env) -> nn.Module:
        raise NotImplementedError()

    def run(self):
        device = T.device("cuda" if T.cuda.is_available() else "cpu")  # type: ignore
        cpu = T.device("cpu")  # type: ignore

        env = spawn_env(
            self.config.environment,
            self.config.game_speed,
            spatial_dim=self.config.screen_size,
            rank=-1,
        )
        model = self.get_model(env).to(device)
        wandb.watch(model, log="all")

        max_reward = 0

        if self.config.use_ppo:
            previous_model = self.get_model(env).to(device)
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

        eps_rewards = np.zeros((self.config.num_workers, 1))
        reward_list = [0]

        step_n = 0
        with tqdm(range(self.config.total_steps), unit="steps", leave=True) as pbar:
            while step_n < self.config.total_steps:
                # play
                transitions = []

                for i in tqdm(
                    range(self.config.unroll_length), unit="steps", leave=False
                ):
                    screen = screen.to(device)
                    minimap = minimap.to(device)
                    non_spatial = non_spatial.to(device)
                    action_mask = action_mask.to(device)
                    logits, value = model(screen, minimap, non_spatial, action_mask)

                    dist = MultiCategorical(*logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(*action)

                    if self.config.use_ppo:
                        with T.no_grad():
                            old_logits, _old_value = previous_model(screen, minimap, non_spatial, action_mask)  # type: ignore
                            old_log_prob = MultiCategorical(*old_logits).log_prob(
                                *action
                            )
                    else:
                        # we won't use this
                        old_log_prob = log_prob

                    entropy = dist.entropy()
                    action = self.interpret_action(action)

                    (
                        screen,
                        minimap,
                        non_spatial,
                        reward,
                        done,
                        action_mask,
                    ) = self.env.step(action.detach().to(cpu))
                    screen = screen.to(device)
                    minimap = minimap.to(device)
                    non_spatial = non_spatial.to(device)
                    action_mask = action_mask.to(device)

                    with torch.no_grad():
                        (
                            _,
                            next_value,
                        ) = model(screen, minimap, non_spatial, action_mask)

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
                    advantages = torch.zeros_like(values).to(device)
                    lastgaelam = 0.0
                    for t in reversed(range(len(values))):
                        if t == len(values) - 1:
                            nextnonterminal = 1.0 - dones[-1].float()
                            nextvalues = next_values[-1]
                        else:
                            nextnonterminal = 1.0 - dones[t+1].float()
                            nextvalues = next_values[t]
                        delta = rewards[t] + self.config.reward_decay * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + self.config.reward_decay * self.config.trace_decay * nextnonterminal * lastgaelam
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards).to(device)
                    for t in reversed(range(len(values))):
                        if t == len(values) - 1:
                            nextnonterminal = 1.0 - dones[-1].float()
                            next_return = values[-1]
                        else:
                            nextnonterminal = 1.0 - dones[t+1].float()
                            next_return = returns[t+1]
                        returns[t] = rewards[t] + self.config.reward_decay * nextnonterminal * next_return
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
                    policy_loss = T.max(
                        surrogate_objective * -advantages.detach(),
                        clipped_surrogate_objective * -advantages.detach(),
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
                            environment_name=self.config.environment, step_n=step_n
                        )
                    )
                    wandb.log_artifact(
                        self.last_model(
                            model,
                            environment_name=self.config.environment, step_n=step_n
                        )
                    )

                wandb.log(metrics, step=step_n, commit=True)

                cycles += 1

                _played_steps = self.config.unroll_length * self.config.num_workers
                pbar.update(_played_steps)
                step_n += _played_steps

        wandb.finish()
        self.env.stop()
