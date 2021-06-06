from typing import List, Tuple
from betastar.envs.env import PySC2Env
from torch.distributions.categorical import Categorical
import numpy
import wandb
import numpy as np
from absl import flags
from betastar.agents import base_agent
from tqdm import tqdm
import torch as T
from torch import nn

FLAGS = flags.FLAGS
FLAGS([__file__])


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
        return T.cat([a, b, c]).flatten()


class ActorCritic(nn.Module):
    def __init__(
        self,
        env: PySC2Env,
        encoded_size: int = 128,
    ) -> None:
        super().__init__()
        screen_channels: int = env.feature_original_shapes[0][0] # type: ignore
        minimap_channels: int = env.feature_original_shapes[1][0] # type: ignore
        non_spatial_channels: int = env.feature_original_shapes[2][0] # type: ignore

        self.pysc2_env = env
        self.encoder = Encoder(
            encoded_size=encoded_size,
            screen_channels=screen_channels,
            minimap_channels=minimap_channels,
            non_spatial_channels=non_spatial_channels,
        )

        self.actor = nn.Sequential(
            nn.Linear(encoded_size * 3, self.pysc2_env.action_space.nvec.sum()),  # type: ignore
        )

    def encode(self, screens, minimaps, non_spatials):
        return self.encoder(screens, minimaps, non_spatials)

    def act(self, latent) -> np.ndarray:
        logits = self._masked_logits(self.actor(latent))
        categoricals = self.discrete_categorical_distributions(logits)
        return T.stack([categorical.sample() for categorical in categoricals]).numpy()

    def _masked_logits(self, logits: T.Tensor) -> T.Tensor:
        """
        Filters the logits according to the environment's action mask.
        This ensures that actions that are not currently available will
        never be sampled.
        """
        action_mask = T.from_numpy(self.pysc2_env.action_mask).bool()
        return T.where(action_mask, logits, T.tensor(-1e8))

    def discrete_categorical_distributions(self, logits: T.Tensor) -> List[Categorical]:
        """
        Split the flat logits into the subsets that represent the many
        categorical distributions (actions, screen coordinates, minimap coordinates, etc...).
        Returns a Categorical object for each of those.
        """
        split_logits = T.split(logits, self.pysc2_env.action_space.nvec.tolist())  # type: ignore
        return [Categorical(logits=logits) for logits in split_logits]


class A3C(base_agent.BaseAgent):
    def __init__(self, config: wandb.Config) -> None:
        super().__init__(config)

    def run(self):
        device = T.device('cpu')
        model = ActorCritic(env=self.env)  # type: ignore
        model.to(device)
        with tqdm(range(self.episodes), unit="episodes") as pbar:
            for episode in pbar:
                observation = self.env.reset()
                done = False
                t = 0
                while not done:
                    self.env.render()
                    screen, minimap, non_spatial = observation
                    latent = model.encode( 
                        screen.unsqueeze(0).to(device),
                        minimap.unsqueeze(0).to(device),
                        non_spatial.unsqueeze(0).to(device)
                    )
                    action = model.act(latent)
                    observation, reward, done, info = self.env.step(action)
                    pbar.set_postfix(steps=str(t))
                    t += 1

        self.env.close()
