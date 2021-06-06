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
from torch.optim import Adam
from tqdm import tqdm
from tqdm_multiprocess import TqdmMultiProcessPool

FLAGS = flags.FLAGS
FLAGS([__file__])

Step = Tuple[Observation, Action, Reward, Value]


@dataclass
class Episode:
    states: List[Observation] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    rewards: List[Reward] = field(default_factory=list)
    values: List[Value] = field(default_factory=list)
    count: int = 0

    def __repr__(self) -> str:
        return f"#({self.count} steps, {sum(self.rewards)} reward)"

    def __len__(self):
        return self.count

    def lookup(self, idxs):
        return [self[idx] for idx in idxs]

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.rewards[idx], self.values[idx]

    def add(
        self, observation: Observation, action: Action, reward: Reward, value: Value
    ):
        self.states.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
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
    ):
        super().__init__()

        self.screen = nn.Sequential(
            Scale(1 / 255),
            nn.Conv2d(screen_channels, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, encoded_size),
        )

        self.minimap = nn.Sequential(
            Scale(1 / 255),
            nn.Conv2d(minimap_channels, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, encoded_size),
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
        )

        self.actor = nn.Sequential(
            nn.Linear(encoded_size * 3, self.action_space.sum()),  # type: ignore
        )

        self.critic = nn.Sequential(nn.Linear(encoded_size * 3, 1), nn.Tanh())

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
        return self.env.reset()

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
    rank: int,
    tqdm_func,
    global_tqdm,
) -> Tuple[List[Episode], List[float]]:
    player = ResilientPlayer(state_dict, environment, game_speed, rank==0)

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
                        actions, values = player.model(
                            screen.unsqueeze(0),
                            minimap.unsqueeze(0),
                            non_spatial.unsqueeze(0),
                            action_mask=player.env.action_mask,
                        )
                        action = actions[0]
                        value = values[0]
                    observation, reward, done, _info = player.env.step(action)

                    value = 0.0 if done else value.item()

                    episode_reward += reward

                    episode.add(observation, action, reward, value)
                episodes.append(episode)
                episode_rewards.append(episode_reward)

                observation = player.env.reset()
                success = True
                global_tqdm.update()
            except protocol.ConnectionError:
                print("FUCK, again for FFFFFFUCKS sake")
                time.sleep(2) # deep breath
                observation = player.restart()

    player.stop()

    return episodes, episode_rewards

def dataloader(episodes: List[Episode], batch_size: int) -> List[List[Step]]:
    episodes = [episode for episode in episodes if len(episode) >= batch_size]

    batches = []

    for episode in episodes:
        idxs = np.arange(len(episode))
        chunks_n = math.floor(len(episode) / batch_size)

        for broad_chunk in np.array_split(idxs, chunks_n):
            batches.append(episode.lookup(broad_chunk[:batch_size]))
            batches.append(episode.lookup(broad_chunk[::-1][:batch_size][::-1]))

    shuffle(batches)
    return batches


def compute_returns(
    rewards: List[float],
    last_q_value: float,
    reward_decay: float,
):
    rews = Tensor(rewards).flip(0).view(-1)
    discounted_rewards = []
    R = last_q_value
    for r in range(rews.shape[0]):
        R = rews[r] + reward_decay * R
        discounted_rewards.append(R)
    return F.normalize(T.stack(discounted_rewards).view(-1), dim=0)


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
        device = T.device("cuda" if T.cuda.is_available() else "cpu")

        env = spawn_env(self.config.environment, self.config.game_speed)
        self.model = ActorCritic(env=env).to(device)
        self.model.train()
        env.close()

        opt = Adam(
            self.model.parameters(), lr=self.config.learning_rate, betas=(0.92, 0.999)
        )

        wandb.watch(self.model, log="all")

        for epoch in range(self.config.epochs):
            # 1. play some episodes

            # EMERGENCY WARNING: Uncomment this piece of code if things break
            # play_episodes(
            #     self.model.state_dict(),  # type: ignore
            #     self.config.environment,
            #     self.config.game_speed,
            #     self.config.episodes_per_epoch,
            #     0,
            #     tqdm,
            #     tqdm,
            #     )
            # break

            episodes: List[Episode] = []

            def error_callback(result):
                print("Error!")

            def done_callback(result):
                print("Done. Result: ", result)

            pool = TqdmMultiProcessPool(self.config.num_workers)
            initial_tasks = [
                (
                    play_episodes,
                    (
                        self.model.state_dict(),
                        self.config.environment,
                        self.config.game_speed,
                        self.config.episodes_per_epoch,
                        rank,
                    ),
                )
                for rank in range(self.config.num_workers)
            ]

            episodes: List[Episode] = []
            episode_rewards: List[float] = []
            with tqdm(
                total=self.config.episodes_per_epoch, dynamic_ncols=True
            ) as global_progress:
                global_progress.set_description("global")
                results: List[Tuple[List[Episode], List[float]]] = pool.map(
                    global_progress, initial_tasks, error_callback, done_callback
                )  # type: ignore
                for eps in results:
                    episodes.extend(eps[0])
                    episode_rewards.extend(eps[1])

            actor_losses = []
            critic_losses = []
            real_losses = []

            for batch in dataloader(episodes, batch_size=20):
                screens = T.stack([step[0][0] for step in batch]).to(device)
                minimaps = T.stack([step[0][1] for step in batch]).to(device)
                non_spatials = T.stack([step[0][2] for step in batch]).to(device)
                actions = T.stack([T.from_numpy(step[1]) for step in batch], dim=0).to(
                    device
                )

                latent = self.model.encoder(screens, minimaps, non_spatials)
                logits = self.model.actor(
                    latent
                )  # no need to mask logits since we're not acting
                categoricals = self.model.discrete_categorical_distributions(logits)
                log_probs = T.stack(
                    [
                        categorical.log_prob(a)
                        for a, categorical in zip(actions.transpose(0, 1), categoricals)
                    ],
                    dim=1,
                )

                returns = compute_returns(
                    [step[2] for step in batch],
                    [step[3] for step in batch][-1],
                    self.config.reward_decay,
                )
                values = self.model.critic(latent.detach()).flip(0).view(-1)
                advantages = returns - values.detach()
                actor_loss = (-1 * log_probs.sum(dim=1) * advantages).sum()
                critic_loss = F.mse_loss(values, returns)
                loss = actor_loss + critic_loss

                opt.zero_grad()
                loss.backward()
                opt.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                real_losses.append(loss.item())

            metrics = {
                "episode_reward/mean": np.array(episode_rewards).mean(),
                "loss/actor": np.array(actor_losses).mean(),
                "loss/critic": np.array(critic_losses).mean(),
                "loss/real": np.array(real_losses).mean(),
                "epoch": epoch,
            }
            wandb.log(
                {
                    **metrics,
                    "video": self.last_video(epoch)
                },
                step=epoch
            )
            wandb.log_artifact(self.last_replay(map_name=self.config.environment, epoch=epoch))
            print(metrics)
