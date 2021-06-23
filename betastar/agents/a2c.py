from typing import List, Tuple

import numpy as np
import torch as T
import wandb
from betastar.agents import base_agent
from betastar.data import Episode, EpisodeDataset, TrajectoryDataset, collate
from betastar.envs.env import (
    Action,
    ActionMask,
    Observation,
    PySC2Env,
    Reward,
    Value,
    spawn_env,
)
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


class Encoder(nn.Module):
    def __init__(
        self,
        screen_channels: int = 5,
        minimap_channels: int = 4,
        non_spatial_channels: int = 34,
        encoded_size: int = 128,
        spatial_dim: int = 64,
    ):
        conv_out_width = int(spatial_dim / 2 - 2)

        super().__init__()

        self.screen = nn.Sequential(
            Scale(1 / 255),
            nn.Conv2d(screen_channels, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * conv_out_width * conv_out_width, encoded_size),
            nn.ReLU(),
        )

        self.minimap = nn.Sequential(
            Scale(1 / 255),
            nn.Conv2d(minimap_channels, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * conv_out_width * conv_out_width, encoded_size),
            nn.ReLU(),
        )

        self.non_spatial = nn.Sequential(
            nn.Linear(non_spatial_channels, encoded_size), nn.ReLU()
        )

    def forward(self, screens, minimaps, non_spatials):
        a = self.screen(screens)
        b = self.minimap(minimaps)
        c = self.non_spatial(non_spatials)
        return T.cat([a, b, c]).view(screens.shape[0], -1)


class ActorCritic(nn.Module):
    def __init__(
        self, env: PySC2Env, encoded_size: int = 128, hidden_size: int = 128
    ) -> None:
        super().__init__()
        screen_channels: int = env.feature_original_shapes[0][0]  # type: ignore
        minimap_channels: int = env.feature_original_shapes[1][0]  # type: ignore
        non_spatial_channels: int = env.feature_original_shapes[2][0]  # type: ignore

        self.action_space = env.action_space.nvec.copy()  # type: ignore

        self.encoder = Encoder(
            encoded_size=encoded_size,
            screen_channels=screen_channels,
            minimap_channels=minimap_channels,
            non_spatial_channels=non_spatial_channels,
            spatial_dim=env.spatial_dim,
        )

        self.backbone = nn.Sequential(
            nn.Linear(encoded_size * 3, hidden_size), nn.ReLU()
        )

        self.actor = nn.Linear(hidden_size, self.action_space.sum())  # type: ignore

        self.critic = nn.Linear(hidden_size, 1)

    def forward(
        self, screens, minimaps, non_spatials, action_mask: ActionMask
    ) -> Tuple[Action, Value]:
        latent = self.encode(screens, minimaps, non_spatials)
        return self.act(latent, action_mask=action_mask), self.critic(latent.detach())

    def encode(self, screens, minimaps, non_spatials) -> Tensor:
        return self.backbone(self.encoder(screens, minimaps, non_spatials))

    def act(self, latent, action_mask: ActionMask) -> Action:
        logits = self._masked_logits(self.actor(latent), action_mask)
        categoricals = self.discrete_categorical_distributions(logits)
        return T.stack([categorical.sample() for categorical in categoricals], dim=1)

    def _masked_logits(self, logits: Tensor, action_mask: ActionMask) -> Tensor:
        """
        Filters the logits according to the environment's action mask.
        This ensures that actions that are not currently available will
        never be sampled.
        """
        return T.where(action_mask.bool(), logits, T.tensor(-1e8))

    def discrete_categorical_distributions(self, logits: Tensor) -> List[Categorical]:
        """
        Split the flat logits into the subsets that represent the many
        categorical distributions (actions, screen coordinates, minimap coordinates, etc...).
        Returns a Categorical object for each of those.
        """
        split_logits = T.split(logits, self.action_space.tolist(), dim=1)  # type: ignore
        return [Categorical(logits=logits) for logits in split_logits]


def compute_trajectory_returns(rewards: Tensor, values: Tensor, config: wandb.Config):
    returns = []
    R = values[-1]
    for r in reversed(rewards):
        R = r + R * config.reward_decay
        returns.insert(0, R)

    returns = T.tensor(returns)
    if config.normalize_returns:
        return (returns - returns.mean()) / (
            returns.std() + np.finfo(np.float32).eps.item()
        )
    else:
        return returns


def compute_traj_returns(
    rewards: Tensor, values: Tensor, dones: Tensor, config: wandb.Config
):
    trajectory_rewards = rewards.split(config.traj_length)
    trajectory_values = values.split(config.traj_length)

    batch_returns = []
    for (rewards, values) in zip(trajectory_rewards, trajectory_values):
        batch_returns.append(compute_trajectory_returns(rewards, values, config))

    return T.cat(batch_returns)


def compute_episode_returns(rewards: Tensor, config: wandb.Config):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * config.reward_decay
        returns.insert(0, R)

    returns = T.tensor(returns)
    if config.normalize_returns:
        return (returns - returns.mean()) / (
            returns.std() + np.finfo(np.float32).eps.item()
        )
    else:
        return returns


def compute_advantages(
    rewards: Tensor, values: Tensor, next_value: float, config: wandb.Config
):
    advantages = []
    advantage = 0

    for r, v in zip(reversed(rewards), reversed(values)):
        td_error = r + next_value * config.reward_decay - v
        advantage = td_error + advantage * config.reward_decay * config.gae_lambda
        next_value = v
        advantages.insert(0, advantage)

    advantages = T.stack(advantages)
    if config.normalize_advantages:
        return (advantages - advantages.mean()) / (
            advantages.std() + np.finfo(np.float32).eps.item()
        )
    else:
        return advantages


def compute_returns(
    rewards: Tensor, values: Tensor, dones: Tensor, config: wandb.Config
):
    episode_limits = T.where(dones == True)[0] + 1
    episodes_rewards = rewards.tensor_split(episode_limits, dim=0)[:-1]
    episodes_values = values.tensor_split(episode_limits, dim=0)[:-1]

    batch_returns = []
    for (rewards, values) in zip(episodes_rewards, episodes_values):
        batch_returns.append(compute_episode_returns(rewards, config))

    return T.cat(batch_returns)


def compute_traj_gae(
    rewards: Tensor, values: Tensor, next_values: Tensor, config: wandb.Config
):
    trajectory_rewards = rewards.split(config.traj_length)
    trajectory_values = values.split(config.traj_length)
    trajectory_next_values = next_values.split(config.traj_length)

    batch_advantages = []
    for (rewards, values, next_values) in zip(
        trajectory_rewards, trajectory_values, trajectory_next_values
    ):
        batch_advantages.append(compute_advantages(rewards, values, next_values[-1].item(), config))  # type: ignore

    return T.cat(batch_advantages)


def compute_gae(rewards: Tensor, values: Tensor, dones: Tensor, config: wandb.Config):
    episode_limits = T.where(dones == True)[0] + 1
    episodes_rewards = rewards.tensor_split(episode_limits, dim=0)[:-1]
    episodes_values = values.tensor_split(episode_limits, dim=0)[:-1]

    batch_advantages = []
    for (rewards, values) in zip(episodes_rewards, episodes_values):
        batch_advantages.append(compute_advantages(rewards, values, 0, config))

    return T.cat(batch_advantages)


class A2CPlayer(Player):
    def __init__(self, model: ActorCritic) -> None:
        super(A2CPlayer).__init__()
        self.model = model

    def act(
        self, screen: Tensor, minimap: Tensor, non_spatial: Tensor, action_mask: Tensor
    ) -> Tensor:
        latents = self.model.encode(screen, minimap, non_spatial)
        return self.model.act(
            latents,
            action_mask=action_mask,
        )

    def evaluate(self, screen: Tensor, minimap: Tensor, non_spatial: Tensor) -> Tensor:
        latents = self.model.encode(screen, minimap, non_spatial)
        return self.model.critic(latents)

    def reload_from(self, model: nn.Module):
        self.model.load_state_dict(model.state_dict())  # type: ignore
        self.model.eval()


class A2C(base_agent.BaseAgent):
    def dataloader(self, episodes: List[Episode]) -> DataLoader:
        if self.config.traj_length > 0:
            return DataLoader(
                TrajectoryDataset(episodes),
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=collate,
            )
        else:
            return DataLoader(
                EpisodeDataset(episodes),
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate,
            )

    def run(self):
        device = T.device("cuda" if T.cuda.is_available() else "cpu")  # type: ignore

        env = spawn_env(self.config.environment, self.config.game_speed, monitor=False)
        model = ActorCritic(env=env).to(device)
        model.train()

        player = A2CPlayer(ActorCritic(env=env))

        env.close()

        opt = Adam(
            model.parameters(), lr=self.config.learning_rate, betas=(0.92, 0.999)
        )

        wandb.watch(model, log="all")

        for epoch in trange(self.config.epochs, unit="epochs"):
            player.reload_from(model)

            episodes, episode_rewards = self.play(player)

            real_losses = []
            actor_losses = []
            critic_losses = []
            entropies = []

            with tqdm(
                self.dataloader(episodes), unit="batches", leave=False
            ) as batches:
                for (
                    screens,
                    minimaps,
                    non_spatials,
                    actions,
                    rewards,
                    _values,
                    next_values,
                    action_masks,
                    dones,
                ) in batches:
                    latents = model.encode(screens, minimaps, non_spatials)
                    logits = T.where(
                        action_masks.bool(), model.actor(latents), T.tensor(-1e8)
                    )

                    values = model.critic(latents.detach())

                    categoricals = model.discrete_categorical_distributions(logits)
                    log_probs = T.stack(
                        [
                            categorical.log_prob(a)
                            for a, categorical in zip(
                                actions.transpose(0, 1), categoricals
                            )
                        ],
                        dim=1,
                    )

                    entropy = T.stack(
                        [categorical.entropy().mean() for categorical in categoricals]
                    ).mean()

                    if self.config.traj_length > 0:
                        returns = compute_traj_returns(
                            rewards,
                            values,
                            dones,
                            self.config,
                        )
                    else:
                        returns = compute_returns(rewards, values, dones, self.config)
                    advantage = returns - values

                    if self.config.use_gae:
                        if self.config.traj_length > 0:
                            pi_advantage = compute_traj_gae(
                                rewards, values, next_values, self.config
                            )
                        else:
                            pi_advantage = compute_gae(
                                rewards, values, dones, self.config
                            )
                    else:
                        pi_advantage = advantage

                    values = model.critic(latents.detach()).squeeze()
                    actor_loss = (-log_probs.sum(dim=1) * pi_advantage.detach()).mean()
                    critic_loss = advantage.pow(2).mean() * self.config.critic_coeff
                    entropy_loss = entropy * self.config.entropy_coeff
                    loss = actor_loss + critic_loss - entropy_loss

                    batches.set_postfix(
                        {
                            "loss": loss.item(),
                            "al": actor_loss.item(),
                            "cl": critic_loss.item(),
                            "ent": entropy_loss.item(),
                        }
                    )

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    real_losses.append(loss.item())
                    actor_losses.append(actor_loss.item())
                    critic_losses.append(critic_loss.item())
                    entropies.append(entropy_loss.item())

            metrics = {
                "episode_reward/mean": np.array(episode_rewards).mean(),
                "episode_reward/max": np.array(episode_rewards).max(),
                "loss/real": np.array(real_losses).mean(),
                "loss/actor": np.array(actor_losses).mean(),
                "loss/critic": np.array(critic_losses).mean(),
                "loss/entropy": np.array(entropies).mean(),
                "epoch": epoch,
            }
            wandb.log(
                {**metrics, "video": self.last_video(epoch)},
            )
            wandb.log_artifact(
                self.last_replay(map_name=self.config.environment, epoch=epoch)
            )

        self.env.stop()
