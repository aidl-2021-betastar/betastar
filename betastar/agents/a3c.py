from typing import Tuple
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

# SCREEN FEATURES (84x84)
feature_screen = ['height_map', 'visibility_map', 'creep', 'power', 'player_id', 'player_relative', 'unit_type', 'selected', 'unit_hit_points', 'unit_hit_points_ratio', 'unit_energy', 'unit_energy_ratio', 'unit_shields', 'unit_shields_ratio', 'unit_density', 'unit_density_aa', 'effects', 'hallucinations', 'cloaked', 'blip', 'buffs', 'buff_duration', 'active', 'build_progress', 'pathable', 'buildable', 'placeholder']

def hydrate_observation(observation: np.ndarray, env: PySC2Env, device: T.device) -> Tuple[T.Tensor, T.Tensor, T.Tensor]:
    screen, minimap, non_spatials = T.split(T.from_numpy(observation).float(), env.feature_flatten_shapes)
    screen_shape, minimap_shape, non_spatials_shape = env.feature_original_shapes

    return screen.reshape(screen_shape).to(device), minimap.reshape(minimap_shape).to(device), non_spatials.reshape(non_spatials_shape).to(device)

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

class ActorCritic(nn.Module):
    def __init__(self, env: PySC2Env, screen_channels: int = 5, minimap_channels: int = 4, non_spatial_channels: int = 34) -> None:
        super().__init__()
        self.pysc2_env = env
        self.screen = nn.Sequential(
            Scale(1/255),
            nn.Conv2d(screen_channels, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*6*6, 128)
        )

        self.minimap = nn.Sequential(
            Scale(1/255),
            nn.Conv2d(minimap_channels, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*6*6, 128)
        )

        self.non_spatial = nn.Sequential(
            nn.Linear(non_spatial_channels, 128),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(128*3, self.pysc2_env.action_space.nvec.sum()), # type: ignore
        )

    def forward(self, screens, minimaps, non_spatials):
        a = self.screen(screens)
        b = self.minimap(minimaps)
        c = self.non_spatial(non_spatials)
        return T.cat([a,b,c]).flatten()

    def act(self, screen, minimap, non_spatials):
        input = self.forward(screen, minimap, non_spatials)
        logits = self.actor(input)

        action_mask = T.from_numpy(self.pysc2_env.action_mask).bool()
        masked_logits = T.where(action_mask, logits, T.tensor(-1e+8))

        split_logits = T.split(masked_logits, self.pysc2_env.action_space.nvec.tolist()) # type: ignore

        multi_categoricals = [Categorical(logits=logits) for logits in split_logits]

        action = T.stack([categorical.sample() for categorical in multi_categoricals])
        return action





    

class A3C(base_agent.BaseAgent):
    def __init__(self, config: wandb.Config) -> None:
        super().__init__(config)

    def run(self):
        model = ActorCritic(env=self.env) # type: ignore
        with tqdm(range(self.episodes), unit="episodes") as pbar:
            for episode in pbar:
                observation = self.env.reset()
                screen, minimap, non_spatial = hydrate_observation(observation, self.env, T.device('cpu')) # type: ignore
                done = False
                t = 0
                while not done:
                    self.env.render()
                    action = model.act(screen.unsqueeze(0), minimap.unsqueeze(0), non_spatial.unsqueeze(0))
                    observation, reward, done, info = self.env.step(action.numpy())
                    pbar.set_postfix(steps=str(t))
                    t += 1

        self.env.close()