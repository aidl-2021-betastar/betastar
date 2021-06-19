from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import betastar.envs
import numpy as np
import torch as T
import wandb
from absl import flags
from betastar.agents import base_agent
from betastar.agents.worker import MultiProcEnv
from betastar.envs.env import (
    Action,
    ActionMask,
    Observation,
    PySC2Env,
    Reward,
    Value,
    spawn_env,
)
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

FLAGS = flags.FLAGS
FLAGS([__file__])

Step = Tuple[Observation, Action, Reward, Value]


@dataclass
class Episode:
    states: List[Observation] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    rewards: List[Reward] = field(default_factory=list)
    values: List[Value] = field(default_factory=list)
    action_masks: List[ActionMask] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    count: int = 0

    def __repr__(self) -> str:
        return f"#({self.count} steps, {sum(self.rewards)} reward)"

    def __len__(self):
        return self.count

    def lookup(self, idxs):
        return [self[idx] for idx in idxs]

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.values[idx],
            self.action_masks[idx],
            self.dones[idx],
        )

    def add(
        self,
        observation: Observation,
        action: Action,
        reward: Reward,
        value: Value,
        action_mask: Tensor,
        done: bool,
    ):
        self.states.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.action_masks.append(action_mask)
        self.dones.append(done)
        self.count += 1


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
        )

        self.minimap = nn.Sequential(
            Scale(1 / 255),
            nn.Conv2d(minimap_channels, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * conv_out_width * conv_out_width, encoded_size),
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
        self,
        env: PySC2Env,
        encoded_size: int = 128,
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

        self.actor = nn.Sequential(
            nn.Linear(encoded_size * 3, self.action_space.sum()),  # type: ignore
        )

        self.critic = nn.Linear(encoded_size * 3, 1)

    def forward(
        self, screens, minimaps, non_spatials, action_mask: ActionMask
    ) -> Tuple[Action, Value]:
        latent = self.encoder(screens, minimaps, non_spatials)
        return self.act(latent, action_mask=action_mask), self.critic(latent.detach())

    def encode(self, screens, minimaps, non_spatials) -> Tensor:
        return self.encoder(screens, minimaps, non_spatials)

    def act(self, latent, action_mask: ActionMask):
        logits = self._masked_logits(self.actor(latent), action_mask)
        categoricals = self.discrete_categorical_distributions(logits)
        return T.stack(
            [categorical.sample() for categorical in categoricals], dim=1
        ).numpy()

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


class EpisodeDataset(Dataset):
    def __init__(self, episodes: List[Episode]) -> None:
        super().__init__()
        self.episodes = episodes

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]


def compute_episode_returns(rewards: Tensor, config: wandb.Config):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * config.reward_decay
        returns.insert(0, R)

    returns = T.tensor(returns)
    return (returns - returns.mean()) / (
        returns.std() + np.finfo(np.float32).eps.item()
    )


def compute_advantages(rewards: Tensor, values: Tensor, config: wandb.Config):
    advantages = []
    advantage = 0
    next_value = 0

    for r, v in zip(reversed(rewards), reversed(values)):
        td_error = r + next_value * config.reward_decay - v
        advantage = td_error + advantage * config.reward_decay * config.gae_lambda
        next_value = v
        advantages.insert(0, advantage)

    advantages = T.stack(advantages)
    return (advantages - advantages.mean()) / (
        advantages.std() + np.finfo(np.float32).eps.item()
    )


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


def compute_gae(rewards: Tensor, values: Tensor, dones: Tensor, config: wandb.Config):
    episode_limits = T.where(dones == True)[0] + 1
    episodes_rewards = rewards.tensor_split(episode_limits, dim=0)[:-1]
    episodes_values = values.tensor_split(episode_limits, dim=0)[:-1]

    batch_advantages = []
    for (rewards, values) in zip(episodes_rewards, episodes_values):
        batch_advantages.append(compute_advantages(rewards, values, config))

    return T.cat(batch_advantages)


def collate(batch):
    screens = T.stack([step[0][0] for trajectory in batch for step in trajectory])
    minimaps = T.stack([step[0][1] for trajectory in batch for step in trajectory])
    non_spatials = T.stack([step[0][2] for trajectory in batch for step in trajectory])
    actions = T.stack(
        [T.from_numpy(step[1]) for trajectory in batch for step in trajectory], dim=0
    )
    rewards = T.tensor([step[2] for trajectory in batch for step in trajectory])
    values = T.tensor([step[3] for trajectory in batch for step in trajectory])
    action_masks = T.stack([step[4] for trajectory in batch for step in trajectory])
    dones = T.tensor([step[5] for trajectory in batch for step in trajectory])

    return (
        screens,
        minimaps,
        non_spatials,
        actions,
        rewards,
        values,
        action_masks,
        dones,
    )


class A3C(base_agent.BaseAgent):
    def __init__(self, config: wandb.Config) -> None:
        super().__init__(config)

    def last_video(self, epoch: int) -> wandb.Video:
        videos = list(Path("/tmp/betastar").glob("*.mp4"))
        videos.sort()
        return wandb.Video(str(videos[-1]), f"epoch={epoch}")

    def last_replay(self, map_name: str, epoch: int) -> wandb.Artifact:
        artifact = wandb.Artifact(
            name=f"{map_name}.{epoch}",
            type="replay",
            metadata={
                "map": map_name,
                "epoch": epoch,
            },
        )
        replays = list(Path("/tmp/betastar").glob("*.SC2Replay"))
        replays.sort()
        artifact.add_file(str(replays[-1]))
        return artifact

    def run(self):
        device = T.device("cuda" if T.cuda.is_available() else "cpu")  # type: ignore

        env = spawn_env(self.config.environment, self.config.game_speed, monitor=False)
        model = ActorCritic(env=env).to(device)
        model.train()

        player = ActorCritic(env=env)
        player.eval()

        env.close()

        opt = Adam(
            model.parameters(), lr=self.config.learning_rate, betas=(0.92, 0.999)
        )

        wandb.watch(model, log="all")

        env = MultiProcEnv(
            self.config.environment,
            self.config.game_speed,
            count=self.config.num_workers,
        )
        env.start()

        for epoch in trange(self.config.epochs, unit="epochs"):
            player.load_state_dict(model.state_dict())  # type: ignore
            player.eval()
            # 1. play some episodes

            episodes: List[Episode] = []
            episode_rewards: List[float] = []

            screen, minimap, non_spatial, _reward, _done, action_mask = env.reset()

            episodes = []
            episodes_ = [Episode() for n in range(self.config.num_workers)]
            episode_rewards = []
            episode_rewards_ = [0 for n in range(self.config.num_workers)]
            played = 0
            with tqdm(
                range(self.config.batch_size), unit="episodes", leave=False
            ) as pbar:
                while played < self.config.batch_size:
                    latents = player.encode(screen, minimap, non_spatial)
                    actions = player.act(
                        latents,
                        action_mask=action_mask,
                    )
                    values = player.critic(latents)
                    (
                        screen,
                        minimap,
                        non_spatial,
                        rewards,
                        dones,
                        action_mask,
                    ) = env.step(actions)

                    for i in range(self.config.num_workers):
                        episodes_[i].add(
                            (screen[i], minimap[i], non_spatial[i]),
                            actions[i],
                            rewards[i].item(),  # type: ignore
                            values[i],
                            action_mask[i],
                            dones[i].item(),  # type: ignore
                        )
                        episode_rewards_[i] += rewards[i].item()  # type: ignore
                        if dones[i]:
                            episodes.append(episodes_[i])
                            episode_rewards.append(episode_rewards_[i])
                            episodes_[i] = Episode()
                            episode_rewards_[i] = 0

                    # reset done workers and set their next state to the initial observations
                    done_workers = T.where(dones == True)[0]
                    if len(done_workers) > 0:
                        (
                            screen_,
                            minimap_,
                            non_spatial_,
                            _reward,
                            _done,
                            action_mask_,
                        ) = env.reset(only=done_workers.tolist())
                        screen[done_workers] = screen_
                        minimap[done_workers] = minimap_
                        non_spatial[done_workers] = non_spatial_
                        action_mask[done_workers] = action_mask_

                        played += len(done_workers)
                        pbar.update(played)

            real_losses = []

            dataloader = DataLoader(
                EpisodeDataset(episodes),
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate,
            )

            with tqdm(dataloader, unit="batches", leave=False) as batches:
                for (
                    screens,
                    minimaps,
                    non_spatials,
                    actions,
                    rewards,
                    values,
                    action_masks,
                    dones,
                ) in batches:
                    latents = model.encoder(screens, minimaps, non_spatials)
                    logits = T.where(
                        action_masks.bool(), model.actor(latents), T.tensor(-1e8)
                    )

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

                    returns = compute_returns(
                        rewards,
                        values,
                        dones,
                        self.config,
                    )
                    advantage = returns - values

                    if self.config.use_gae:
                        pi_advantage = compute_gae(rewards, values, dones, self.config)
                    else:
                        pi_advantage = advantage

                    values = model.critic(latents.detach()).squeeze()
                    actor_loss = (
                        -log_probs.sum(dim=1) * pi_advantage.detach()
                    ).mean() - entropy * self.config.beta
                    critic_loss = (returns - values).pow(2).mean()
                    loss = actor_loss + critic_loss

                    batches.set_postfix(
                        {
                            "loss": loss.item(),
                            "al": actor_loss.item(),
                            "cl": critic_loss.item(),
                        }
                    )

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    real_losses.append(loss.item())

            metrics = {
                "episode_reward/mean": np.array(episode_rewards).mean(),
                "episode_reward/max": np.array(episode_rewards).max(),
                "loss/real": np.array(real_losses).mean(),
                "epoch": epoch,
            }
            wandb.log(
                {**metrics, "video": self.last_video(epoch)},
            )
            wandb.log_artifact(
                self.last_replay(map_name=self.config.environment, epoch=epoch)
            )

        env.stop()
