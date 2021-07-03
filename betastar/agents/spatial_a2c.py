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


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

class ScreenNet(torch.nn.Module):
    """ Model for movement based mini games in sc2.
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
            assert kwargs["out_channels"] == in_channel, ("input channel must"
                                                          "be the same as out"
                                                          " channels")
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
            self.ResidualConv(32, **res_kwargs)
        )

        self.policy = torch.nn.Sequential(
            torch.nn.Conv2d(32, out_channels=1, kernel_size=1),
            #torch.nn.LayerNorm(in_channel),
            torch.nn.ReLU(),

            torch.nn.Conv2d(1, out_channels=1, kernel_size=1),
            #torch.nn.InstanceNorm2d(1),

            # torch.nn.Linear(32*screen_size*screen_size, 256),
            # torch.nn.LayerNorm(256),
            # torch.nn.ReLU(),
            # torch.nn.Linear(256, 2*screen_size)
        )

        self.value = torch.nn.Sequential(
            torch.nn.Linear(32*screen_size*screen_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

        self.screen_size = screen_size
        gain = torch.nn.init.calculate_gain("relu")

        def param_init(module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight, gain)
                torch.nn.init.zeros_(module.bias)
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.dirac_(module.weight)
                torch.nn.init.zeros_(module.bias) # type: ignore
        self.apply(param_init)

    def forward(self, state):
        encode = self.convnet(state / 255)

        value = self.value(encode.reshape(encode.shape[0], -1).detach())

        logits = self.policy(encode).squeeze(1)

        return logits, value


# class Encoder(nn.Module):
#     def __init__(
#         self,
#         screen_channels: int = 5,
#         minimap_channels: int = 4,
#         non_spatial_channels: int = 34,
#         encoded_size: int = 128,
#         spatial_dim: int = 16,
#     ):
#         conv_out_width = int(spatial_dim / 2 - 2)

#         super().__init__()

#         self.screen = nn.Sequential(
#             Scale(1 / 255),
#             nn.Conv2d(screen_channels, 16, kernel_size=3, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=2),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(32 * conv_out_width * conv_out_width, encoded_size),
#             nn.ReLU(),
#         )

#         # self.minimap = nn.Sequential(
#         #     Scale(1 / 255),
#         #     nn.Conv2d(minimap_channels, 16, kernel_size=3, stride=2),
#         #     nn.ReLU(),
#         #     nn.Conv2d(16, 32, kernel_size=2),
#         #     nn.ReLU(),
#         #     nn.Flatten(),
#         #     nn.Linear(32 * conv_out_width * conv_out_width, encoded_size),
#         #     nn.ReLU(),
#         # )

#         # self.non_spatial = nn.Sequential(
#         #     nn.Linear(non_spatial_channels, encoded_size), nn.ReLU()
#         # )

#     def forward(self, screens, minimaps, non_spatials):
#         a = self.screen(screens)
#         #b = self.minimap(minimaps)
#         #c = self.non_spatial(non_spatials)
#         #return T.cat([a, b, c]).view(screens.shape[0], -1)
#         return a.view(screens.shape[0], -1)


# class ActorCritic(nn.Module):
#     def __init__(
#         self, env: PySC2Env, encoded_size: int = 128, hidden_size: int = 128
#     ) -> None:
#         super().__init__()
#         screen_channels: int = env.feature_original_shapes[0][0]  # type: ignore
#         minimap_channels: int = env.feature_original_shapes[1][0]  # type: ignore
#         non_spatial_channels: int = env.feature_original_shapes[2][0]  # type: ignore


#         if env.action_space.__class__ == spaces.Discrete:
#             self.discrete_action_space = True
#         else:
#             self.discrete_action_space = False
#             self.action_space = env.action_space.nvec.copy()  # type: ignore

#         self.encoder = Encoder(
#             encoded_size=encoded_size,
#             screen_channels=screen_channels,
#             minimap_channels=minimap_channels,
#             non_spatial_channels=non_spatial_channels,
#             spatial_dim=env.spatial_dim,
#         )

#         # self.backbone = nn.Sequential(
#         #     nn.Linear(encoded_size * 3, hidden_size), nn.ReLU()
#         # )

#         # self.actor = nn.Linear(encoded_size * 3, self.action_space.sum())  # type: ignore

#         # self.critic = nn.Linear(encoded_size * 3, 1)
#         self.actor = nn.Linear(encoded_size, env.action_space_length)  # type: ignore

#         self.critic = nn.Linear(encoded_size, 1)

#     def forward(
#         self, screens, minimaps, non_spatials, action_mask: ActionMask
#     ) -> Tuple[Action, Value]:
#         latent = self.encode(screens, minimaps, non_spatials)
#         return self.act(latent, action_mask=action_mask), self.critic(latent.detach())

#     def encode(self, screens, minimaps, non_spatials) -> Tensor:
#         return self.encoder(screens, minimaps, non_spatials)
#         #return self.backbone(self.encoder(screens, minimaps, non_spatials))

#     def act(self, latent, action_mask: ActionMask) -> Action:
#         logits = self._masked_logits(self.actor(latent), action_mask)
#         categoricals = self.discrete_categorical_distributions(logits)
#         return T.stack([categorical.sample() for categorical in categoricals], dim=1)

#     def _masked_logits(self, logits: Tensor, action_mask: ActionMask) -> Tensor:
#         """
#         Filters the logits according to the environment's action mask.
#         This ensures that actions that are not currently available will
#         never be sampled.
#         """
#         return T.where(action_mask.bool(), logits, T.tensor(-1e8))

#     def discrete_categorical_distributions(self, logits: Tensor) -> List[Categorical]:
#         """
#         Split the flat logits into the subsets that represent the many
#         categorical distributions (actions, screen coordinates, minimap coordinates, etc...).
#         Returns a Categorical object for each of those.
#         """
#         if self.discrete_action_space:
#             split_logits = [logits]
#         else:
#             split_logits = T.split(logits, self.action_space.tolist(), dim=1)  # type: ignore
#         return [Categorical(logits=logits) for logits in split_logits]


# def _compute_unroll_returns(rewards: Tensor, bootstrap: float, config: wandb.Config):
#     """
#     Computes returns on a single unroll of arbitrary length.
#     """
#     returns = []
#     R = bootstrap
#     for r in reversed(rewards):
#         R = r + R * config.reward_decay
#         returns.insert(0, R)

#     returns = T.tensor(returns)
#     if config.normalize_returns:
#         return (returns - returns.mean()) / (
#             returns.std() + np.finfo(np.float32).eps.item()
#         )
#     else:
#         return returns


# def compute_returns(
#     rewards: Tensor, next_values: Tensor, dones: Tensor, config: wandb.Config
# ):
#     """
#     Computes returns on a series of unrolls.
#     """
#     unroll_rewards = rewards.split(config.unroll_length)
#     unroll_next_values = next_values.split(config.unroll_length)
#     unroll_dones = dones.split(config.unroll_length)

#     batch_returns = []
#     for (u_rewards, u_next_values, u_dones) in zip(
#         unroll_rewards, unroll_next_values, unroll_dones
#     ):
#         if u_dones[-1]:
#             bootstrap = 0.0
#         else:
#             bootstrap = u_next_values[-1].item()
#         batch_returns.append(_compute_unroll_returns(u_rewards, bootstrap, config))

#     return T.cat(batch_returns)


# # def compute_advantages(
# #     rewards: Tensor, values: Tensor, next_value: float, config: wandb.Config
# # ):
# #     advantages = []
# #     advantage = 0

# #     for r, v in zip(reversed(rewards), reversed(values)):
# #         td_error = r + next_value * config.reward_decay - v
# #         advantage = td_error + advantage * config.reward_decay * config.gae_lambda
# #         next_value = v
# #         advantages.insert(0, advantage)

# #     advantages = T.stack(advantages)
# #     if config.normalize_advantages:
# #         return (advantages - advantages.mean()) / (
# #             advantages.std() + np.finfo(np.float32).eps.item()
# #         )
# #     else:
# #         return advantages


# def compute_gae(
#     rewards: Tensor,
#     values: Tensor,
#     next_values: Tensor,
#     dones: Tensor,
#     config: wandb.Config,
# ) -> Tuple[Tensor, Tensor]:
#     N = len(rewards)
#     advantages = T.zeros_like(rewards).float()
#     lastgaelam = 0.0
#     for t in reversed(range(N)):
#         nextnonterminal = 1.0 - dones[t].item()  # type: ignore
#         nextvalues = next_values[t]

#         delta = (
#             rewards[t] + config.reward_decay * nextvalues * nextnonterminal - values[t]
#         )
#         advantages[t] = lastgaelam = (
#             delta
#             + config.reward_decay * config.gae_lambda * nextnonterminal * lastgaelam
#         )
#     returns = advantages + values

#     if config.normalize_advantages:
#         advantages = (advantages - advantages.mean()) / (
#             advantages.std() + np.finfo(np.float32).eps.item()
#         )
        
#     return advantages, returns


# def compute_gae_batches(
#     rewards: Tensor,
#     values: Tensor,
#     next_values: Tensor,
#     dones: Tensor,
#     config: wandb.Config,
# ):
#     """
#     Computes GAE on a series of unrolls.
#     """
#     unroll_rewards = rewards.split(config.unroll_length)
#     unroll_values = values.split(config.unroll_length)
#     unroll_next_values = next_values.split(config.unroll_length)
#     unroll_dones = dones.split(config.unroll_length)

#     batch_advantages = []
#     batch_returns = []
#     for (u_rewards, u_values, u_next_values, u_dones) in zip(
#         unroll_rewards, unroll_values, unroll_next_values, unroll_dones
#     ):
#         advs, rets = compute_gae(u_rewards, u_values, u_next_values, u_dones, config)
#         batch_advantages.append(advs)
#         batch_returns.append(rets)

#     return T.cat(batch_advantages), T.cat(batch_returns)


# class A2CPlayer(Player):
#     def __init__(self, model: ActorCritic) -> None:
#         super(A2CPlayer).__init__()
#         self.model = model

#     def act(
#         self, screen: Tensor, minimap: Tensor, non_spatial: Tensor, action_mask: Tensor
#     ) -> Tensor:
#         latents = self.model.encode(screen, minimap, non_spatial)
#         return self.model.act(
#             latents,
#             action_mask=action_mask,
#         )

#     def evaluate(self, screen: Tensor, minimap: Tensor, non_spatial: Tensor) -> Tensor:
#         latents = self.model.encode(screen, minimap, non_spatial)
#         return self.model.critic(latents)

#     def reload_from(self, model: nn.Module):
#         self.model.load_state_dict(model.state_dict())  # type: ignore
#         self.model.eval()


class MultiCategorical:
    def __init__(self, *logits):
        self.dists = [Categorical(logits=logit)
                      for logit in logits]

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

        env = spawn_env(self.config.environment, self.config.game_speed, spatial_dim=self.config.screen_size, rank=-1)
        #model = ActorCritic(env=env).to(device)
        model = ScreenNet(env.feature_original_shapes[0][0], self.config.screen_size).to(device)
        wandb.watch(model, log="all")

        max_reward = 0

        # if self.config.use_ppo:
        #     previous_model = ScreenNet(env.feature_original_shapes[0][0], self.config.screen_size).to(device)
        #     #previous_model = ActorCritic(env=env).to(device)
        #     previous_model.load_state_dict(model.state_dict())  # type: ignore

        #player = A2CPlayer(ActorCritic(env=env))

        env.close()

        opt = Adam(
            model.parameters(), lr=self.config.learning_rate, betas=(0.92, 0.999)
        )

        total_cycles = self.config.total_steps // self.config.unroll_length * self.config.num_workers

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

                #player.reload_from(model)

                # play
                transitions = []

                for i in tqdm(range(self.config.unroll_length), unit="steps", leave=False):
                    logits, value = model(screen)
                    dist = Categorical(logits=logits.reshape(logits.shape[0], -1))
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    entropy = dist.entropy()
                    # action == x, y
                    spatial_action = T.from_numpy(np.stack(np.unravel_index(action.cpu().numpy(), (self.config.screen_size, self.config.screen_size)))).T

                    screen, _minimap, _non_spatial, reward, done, _action_mask = self.env.step(spatial_action)
                    screen = screen.to(device)

                    with torch.no_grad():
                        _, next_value, = model(screen)

                    transitions.append((reward, done, log_prob, value.squeeze(), next_value.squeeze(), entropy))

                    for j, d in enumerate(done.flatten()):
                        eps_rewards[j] += reward[j].item()
                        if d == 1:
                            reward_list.append(eps_rewards[j].item())
                            eps_rewards[j] = 0
                    
                    if done.all():
                        break
                
                # learn!
                # all things are whatever x batch_size
                rewards = T.stack([x[0] for x in transitions]).to(device)
                dones = T.stack([x[1] for x in transitions]).to(device)
                log_probs = T.stack([x[2] for x in transitions]).to(device)
                values = T.stack([x[3] for x in transitions]).to(device)
                next_values = T.stack([x[4] for x in transitions]).to(device)
                entropies = T.stack([x[5] for x in transitions]).to(device)

                # compute returns
                R = next_values[-1]
                rets = []
                for i in reversed(range(len(values))):
                    R = R * (1-dones[i].long()) * self.config.reward_decay + rewards[i]
                    rets.append(R)
                returns = T.stack(list(reversed(rets))).to(device)
                if self.config.normalize_returns:
                    returns = (returns - returns.mean()) / (
                        returns.std() + np.finfo(np.float32).eps.item()
                    )

                # compute advantages
                advantages = returns - values
                if self.config.normalize_advantages:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + np.finfo(np.float32).eps.item()
                    )
                
                value_loss = advantages.pow(2).mean() * self.config.critic_coeff
                policy_loss = (-log_probs * advantages.detach()).mean()
                entropy_loss = entropies.mean() * self.config.entropy_coeff

                opt.zero_grad()
                loss = value_loss + policy_loss - entropy_loss
                loss.backward()
                opt.step()

                metrics = {
                    'episode_reward/last10/mean': np.mean(reward_list[-10:]),
                    'episode_reward/last10/max': np.max(reward_list[-10:]),
                    'episode_reward/last50/mean': np.mean(reward_list[-50:]),
                    'episode_reward/last50/max': np.max(reward_list[-50:]),
                    'loss/real': loss.item(),
                    'loss/actor': policy_loss.mean().item(),
                    'loss/critic': value_loss.mean().item(),
                    'loss/entropy': entropies.mean()
                }

                _max_reward = np.max(reward_list)

                if _max_reward > max_reward:
                    wandb.run.summary["episode_reward/max"] = _max_reward
                    max_reward = _max_reward

                if (
                    (cycles > 0 and
                     cycles % self.config.test_interval == 0)
                    or step_n > self.config.total_steps
                ):
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



                # trajectories, trajectory_rewards = self.play(player)

                # real_losses = []
                # actor_losses = []
                # critic_losses = []
                # entropies = []

                # for _update_epoch in tqdm(
                #     range(self.config.update_epochs), unit="epochs", leave=False
                # ):
                #     with tqdm(
                #         self.dataloader(trajectories), unit="batches", leave=False
                #     ) as batches:
                #         for (
                #             screens,
                #             minimaps,
                #             non_spatials,
                #             actions,
                #             rewards,
                #             _values,
                #             next_values,
                #             action_masks,
                #             dones,
                #         ) in batches:
                #             screens = screens.to(device)
                #             minimaps = minimaps.to(device)
                #             non_spatials = non_spatials.to(device)
                #             actions = actions.to(device)
                #             rewards = rewards.to(device)
                #             next_values = next_values.to(device)
                #             action_masks = action_masks.to(device)
                #             dones = dones.to(device)

                #             latents = model.encode(screens, minimaps, non_spatials)
                #             logits = T.where(
                #                 action_masks.bool(),
                #                 model.actor(latents),
                #                 T.tensor(-1e8).to(device),
                #             )

                #             values = model.critic(latents.detach()).squeeze(dim=1)

                #             categoricals = model.discrete_categorical_distributions(
                #                 logits
                #             )
                #             log_probs = T.stack(
                #                 [
                #                     categorical.log_prob(a)
                #                     for a, categorical in zip(
                #                         actions.transpose(0, 1), categoricals
                #                     )
                #                 ],
                #                 dim=1,
                #             )

                #             if self.config.use_ppo:
                #                 old_categoricals = previous_model.discrete_categorical_distributions(logits)  # type: ignore
                #                 old_log_probs = T.stack(
                #                     [
                #                         categorical.log_prob(a)
                #                         for a, categorical in zip(
                #                             actions.transpose(0, 1), old_categoricals
                #                         )
                #                     ],
                #                     dim=1,
                #                 )

                #                 # PPO Clip
                #                 # do we want to aggregate first and then calculate a big ratio?
                #                 # ratio = T.exp(
                #                 #     log_probs.sum(dim=1) - old_log_probs.sum(dim=1)
                #                 # )

                #                 # clipped_ratio = ratio.clamp(
                #                 #     min=1.0 - self.config.clip_range,
                #                 #     max=1.0 + self.config.clip_range,
                #                 # )

                #                 # or do we want to calculate ratios to clip independently, and then aggregate?
                #                 ratio = T.exp(log_probs - old_log_probs)

                #                 clipped_ratio = ratio.clamp(min=1.0 - self.config.clip_range,
                #                                             max=1.0 + self.config.clip_range)

                #                 ratio = ratio.sum(dim=1)
                #                 clipped_ratio = clipped_ratio.sum(dim=1)

                #             entropy = T.stack(
                #                 [
                #                     categorical.entropy().mean()
                #                     for categorical in categoricals
                #                 ]
                #             ).mean()

                #             if self.config.use_gae:
                #                 advantages, returns = compute_gae_batches(
                #                     rewards, values, next_values, dones, self.config
                #                 )
                #             else:
                #                 returns = compute_returns(
                #                     rewards,
                #                     next_values,
                #                     dones,
                #                     self.config,
                #                 )
                #                 advantages = returns - values

                #             if self.config.use_ppo:
                #                 actor_loss = T.min(
                #                     ratio * advantages.detach(),  # type: ignore
                #                     clipped_ratio * advantages.detach(),  # type: ignore
                #                 )  # type: ignore
                #                 actor_loss = (-actor_loss).mean()  # type: ignore
                #             else:
                #                 actor_loss = (
                #                     -log_probs.sum(dim=1) * advantages.detach()
                #                 ).mean()
                #             critic_loss = (
                #                 advantages.pow(2).mean() * self.config.critic_coeff
                #             )
                #             entropy_loss = entropy * self.config.entropy_coeff
                #             loss = actor_loss + critic_loss - entropy_loss

                #             batches.set_postfix(
                #                 {
                #                     "loss": loss.item(),
                #                     "al": actor_loss.item(),
                #                     "cl": critic_loss.item(),
                #                     "ent": entropy_loss.item(),
                #                 }
                #             )

                #             opt.zero_grad()
                #             loss.backward()

                #             if self.config.use_ppo:
                #                 # save model as previous model
                #                 previous_model.load_state_dict(model.state_dict())  # type: ignore

                #             nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                #             opt.step()

                #             real_losses.append(loss.item())
                #             actor_losses.append(actor_loss.item())
                #             critic_losses.append(critic_loss.item())
                #             entropies.append(entropy_loss.item())

                # metrics = {
                #     "loss/real": np.array(real_losses).mean(),
                #     "loss/actor": np.array(actor_losses).mean(),
                #     "loss/critic": np.array(critic_losses).mean(),
                #     "loss/entropy": np.array(entropies).mean(),
                # }
                # if (
                #     (cycles > 0 and
                #      cycles % self.config.test_interval == 0)
                #     or step_n > self.config.total_steps
                # ):
                #     player.reload_from(model)
                #     test_rewards = self.test(player, episodes=5)

                #     _max_reward = np.array(test_rewards).max()
                #     metrics["episode_reward/mean"] = np.array(test_rewards).mean()
                #     metrics["episode_reward/max"] = _max_reward

                #     if _max_reward > max_reward:
                #         wandb.run.summary["episode_reward/max"] = _max_reward
                #         max_reward = _max_reward

                #     metrics["video"] = self.last_video(step_n)
                #     wandb.log_artifact(
                #         self.last_replay(
                #             map_name=self.config.environment, step_n=step_n
                #         )
                #     )

                # wandb.log(metrics, step=step_n, commit=True)

                # cycles += 1

                # _played_steps = sum([len(e) for e in trajectories])
                # pbar.update(_played_steps)
                # step_n += _played_steps

        wandb.finish()
        self.env.stop()
        #self.test_env.stop()
