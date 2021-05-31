import numpy
import wandb
from absl import flags
from betastar.agents import base_agent
from betastar.envs import SC2GameEnv
from tqdm import tqdm
import torch as T
from torch import nn

FLAGS = flags.FLAGS
FLAGS([__file__])

# SCREEN FEATURES (84x84)
feature_screen = ['height_map', 'visibility_map', 'creep', 'power', 'player_id', 'player_relative', 'unit_type', 'selected', 'unit_hit_points', 'unit_hit_points_ratio', 'unit_energy', 'unit_energy_ratio', 'unit_shields', 'unit_shields_ratio', 'unit_density', 'unit_density_aa', 'effects', 'hallucinations', 'cloaked', 'blip', 'buffs', 'buff_duration', 'active', 'build_progress', 'pathable', 'buildable', 'placeholder']

def load_obs(observation, device: T.device):
    screen = T.from_numpy(observation['feature_screen']).to(device)
    minimap = T.from_numpy(observation['feature_minimap']).to(device)
    non_spatials = T.cat([
        T.from_numpy(observation['player']), # (11,)
        T.from_numpy(observation['control_groups']).flatten(), # (20,), originally (10,2)
        T.from_numpy(observation['single_select']).flatten(), # (7,)
    ]).to(device)

class ActorCritic(nn.Module):
    def __init__(self) -> None:
        super(ActorCritic).__init__()
        self.screen = nn.Sequential(
            
        )
    

class A3C(base_agent.BaseAgent):
    def __init__(self, config: wandb.Config) -> None:
        super().__init__(config)

    def run(self):
        with tqdm(range(self.episodes), unit="episodes") as pbar:
            for episode in pbar:
                observation = self.env.reset()
                done = False
                t = 0
                while not done:
                    self.env.render()
                    if self.game_env.__class__ == SC2GameEnv:
                        action = self.select_action()
                    else:
                        action = self.env.action_space.sample()
                    observation, reward, done, info = self.env.step(action)
                    pbar.set_postfix(steps=str(t))
                    t += 1

        self.env.close()

    def select_action(self):
        function_id = numpy.random.choice(self.game_env.available_actions)
        args = [[numpy.random.randint(0, size) for size in arg.sizes]
                for arg in self.sc2env.action_spec()[0].functions[function_id].args]
        return [function_id, *args]
