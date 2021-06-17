from functools import partial
import math
from dataclasses import dataclass, field
from pathlib import Path
from random import shuffle
from typing import List, OrderedDict, Tuple
import time

import betastar.envs
import numpy as np
import torch as T
import torch.nn.functional as F
import wandb
from pysc2.lib import protocol
from absl import flags
from betastar.agents import base_agent
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
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
import torch.multiprocessing as mp

FLAGS = flags.FLAGS
FLAGS([__file__])

Step = Tuple[Observation, Action, Reward, Value]


@dataclass
class Episode:
    states: List[Observation] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    rewards: List[Reward] = field(default_factory=list)
    values: List[Value] = field(default_factory=list)
    action_masks: List[np.ndarray] = field(default_factory=list)
    count: int = 0

    def __repr__(self) -> str:
        return f"#({self.count} steps, {sum(self.rewards)} reward)"

    def __len__(self):
        return self.count

    def lookup(self, idxs):
        return [self[idx] for idx in idxs]

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.rewards[idx], self.values[idx], self.action_masks[idx]

    def add(
        self, observation: Observation, action: Action, reward: Reward, value: Value, action_mask: np.ndarray
    ):
        self.states.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.action_masks.append(action_mask)
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
        spatial_dim: int = 64
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
            spatial_dim=env.spatial_dim
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
        return T.where(T.from_numpy(action_mask).bool(), logits, T.tensor(-1e8))

    def discrete_categorical_distributions(self, logits: Tensor) -> List[Categorical]:
        """
        Split the flat logits into the subsets that represent the many
        categorical distributions (actions, screen coordinates, minimap coordinates, etc...).
        Returns a Categorical object for each of those.
        """
        split_logits = T.split(logits, self.action_space.tolist(), dim=1)  # type: ignore
        return [Categorical(logits=logits) for logits in split_logits]


class ResilientPlayer:
    def __init__(self,
    state_dict: OrderedDict[str, Tensor],
    environment: str, game_speed: int, monitor: bool) -> None:
        self.env = None
        self.state_dict = state_dict
        self.environment = environment
        self.game_speed = game_speed
        self.monitor = monitor

    def start(self) -> Observation:
        self.env = spawn_env(self.environment, self.game_speed, self.monitor)
        self.model = ActorCritic(env=self.env)
        self.model.load_state_dict(self.state_dict)
        self.model.eval()
        while True:
            try:
                return self.env.reset()
            except protocol.ConnectionError:
                print('FFS.........')
                time.sleep(1)

    def stop(self):
        if self.env is not None:
            self.env.close()

    def restart(self) -> Observation:
        self.stop()
        return self.start()
        


def play_episodes(
    state_dict: OrderedDict[str, Tensor],
    environment: str,
    game_speed: int,
    episodes_to_play: int,
    rank: int
) -> Tuple[List[Episode], List[float]]:
    player = ResilientPlayer(state_dict, environment, game_speed, rank==0)
    try:
        episodes = []
        episode_rewards = []

        for _episode_number in range(episodes_to_play):
            observation = player.restart()

            success = False
            while not success:
                try:
                    episode = Episode()
                    episode_reward = 0.0

                    done = False
                    while not done:
                        screen, minimap, non_spatial = observation
                        with T.no_grad():
                            latents = player.model.encode(
                                screen.unsqueeze(0),
                                minimap.unsqueeze(0),
                                non_spatial.unsqueeze(0),
                            )
                            actions = player.model.act(
                                latents,
                                action_mask=player.env.action_mask,
                            )
                            action = actions[0]
                            observation, reward, done, _info = player.env.step(action)

                            # compute value of next state
                            next_screen, next_minimap, next_non_spatial = observation
                            latents = player.model.encode(
                                next_screen.unsqueeze(0),
                                next_minimap.unsqueeze(0),
                                next_non_spatial.unsqueeze(0),
                            )
                            next_value = player.model.critic(latents)[0]

                        value = 0.0 if done else next_value.item()

                        episode_reward += reward

                        episode.add(observation, action, reward, value, player.env.action_mask)
                    episodes.append(episode)
                    episode_rewards.append(episode_reward)

                    observation = player.env.reset()
                    success = True
                    #global_tqdm.update()
                except protocol.ConnectionError:
                    print("FUCK, again for FFFFFFUCKS sake")
                    time.sleep(2) # deep breath
                    observation = player.restart()

        return episodes, episode_rewards
    finally:
        player.stop()

class TrajectoryDataset(Dataset):
    def __init__(self, episodes: List[Episode], n_step: int = 20) -> None:
        super().__init__()
        self.trajectories = []

        for episode in episodes:
            idxs = np.arange(len(episode))
            chunks_n = math.floor(len(episode) / n_step)

            for broad_chunk in np.array_split(idxs, chunks_n):
                self.trajectories.append(episode.lookup(broad_chunk[:n_step]))
                self.trajectories.append(episode.lookup(broad_chunk[::-1][:n_step][::-1]))
    
    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]


def compute_returns(
    rewards: Tensor,
    values: Tensor,
    reward_decay: float,
    n_step: int,
    batch_size: int,
):        
    trajectory_rewards = rewards.split_with_sizes([n_step for x in range(int(len(rewards) / n_step))])
    trajectory_values = values.split_with_sizes([n_step for x in range(int(len(values) / n_step))])
    batched_returns = []
    for (trajectory_reward, trajectory_value) in zip(trajectory_rewards, trajectory_values):
        rews = trajectory_reward.flip(0).view(-1)
        discounted_rewards = []
        R = trajectory_value[-1] # last next_value of the trajectory, = G
        for r in range(rews.shape[0]):
            R = rews[r] + reward_decay * R
            discounted_rewards.append(R)
        batched_returns.append(F.normalize(T.stack(discounted_rewards).view(-1), dim=0))
    return T.cat(batched_returns)


def collate(batch):
    screens = T.stack([step[0][0] for trajectory in batch for step in trajectory])
    minimaps = T.stack([step[0][1] for trajectory in batch for step in trajectory])
    non_spatials = T.stack([step[0][2] for trajectory in batch for step in trajectory])
    actions = T.stack([T.from_numpy(step[1]) for trajectory in batch for step in trajectory], dim=0)
    rewards = T.tensor([step[2] for trajectory in batch for step in trajectory])
    values = T.tensor([step[3] for trajectory in batch for step in trajectory])
    action_masks = T.tensor([step[4] for trajectory in batch for step in trajectory])

    return screens, minimaps, non_spatials, actions, rewards, values, action_masks

class A3C(base_agent.BaseAgent):
    def __init__(self, config: wandb.Config) -> None:
        super().__init__(config)

    def last_video(self, epoch: int) -> wandb.Video:
        videos = list(Path('/tmp/betastar').glob('*.mp4'))
        videos.sort()
        return wandb.Video(str(videos[-1]), f'epoch={epoch}')

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
        ctx = mp.get_context('spawn')
        device = T.device("cuda" if T.cuda.is_available() else "cpu")

        env = spawn_env(self.config.environment, self.config.game_speed, monitor=False)
        self.model = ActorCritic(env=env).to(device)
        self.model.train()
        env.close()

        opt = Adam(
            self.model.parameters(), lr=self.config.learning_rate, betas=(0.92, 0.999)
        )

        wandb.watch(self.model, log="all")

        for epoch in range(self.config.epochs):
            # 1. play some episodes

            episodes: List[Episode] = []
            episode_rewards: List[float] = []

            if self.config.num_workers == 1:
                episodes, episode_rewards = play_episodes(
                    self.model.state_dict(),  # type: ignore
                    self.config.environment,
                    self.config.game_speed,
                    self.config.episodes_per_epoch,
                    0
                    )
            else:
                play = partial(play_episodes,
                self.model.state_dict(),
                self.config.environment,
                self.config.game_speed,
                self.config.episodes_per_epoch
                )
                
                with ctx.Pool(self.config.num_workers) as pool:
                    for eps in list(tqdm(pool.imap(play, range(self.config.num_workers)))):
                        episodes.extend(eps[0])
                        episode_rewards.extend(eps[1])

                    pool.join()

            real_losses = []

            n_step = 200
            batch_size = 2

            dataloader = DataLoader(TrajectoryDataset(episodes, n_step=n_step), batch_size=batch_size, shuffle=True, collate_fn=collate)

            for screens, minimaps, non_spatials, actions, rewards, values, action_masks in dataloader:
                latents = self.model.encoder(screens, minimaps, non_spatials)
                logits = T.where(action_masks.bool(), self.model.actor(latents), T.tensor(-1e8))

                categoricals = self.model.discrete_categorical_distributions(logits)
                log_probs = T.stack(
                    [
                        categorical.log_prob(a)
                        for a, categorical in zip(actions.transpose(0, 1), categoricals)
                    ],
                    dim=1,
                )

                returns = compute_returns(
                    rewards,
                    values,
                    self.config.reward_decay,
                    n_step,
                    batch_size
                )
                # the "proper way"
                # values = self.model.critic(latents.detach()).flip(1).squeeze()
                # advantages = returns - values.detach()
                # actor_loss = (-1 * log_probs.sum(dim=1) * advantages).mean()
                # critic_loss = F.mse_loss(values, returns)
                # loss = actor_loss + critic_loss

                values = self.model.critic(latents.detach()).flip(1).squeeze()
                advantages = returns - values.detach()
                actor_loss = -1 * log_probs.sum(dim=1) * advantages
                critic_loss = advantages.pow(2)
                loss = (actor_loss + critic_loss).mean()

                # log_prob = m.log_prob(a)
                # entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration
                # exp_v = log_prob * td.detach() + 0.005 * entropy
                # a_loss = -exp_v
                # total_loss = (a_loss + c_loss).mean()
                # return total_loss

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
                {
                    **metrics,
                    "video": self.last_video(epoch)
                },
            )
            wandb.log_artifact(self.last_replay(map_name=self.config.environment, epoch=epoch))
            print(metrics)
