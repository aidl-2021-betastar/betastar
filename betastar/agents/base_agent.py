import gym
import random
import wandb
from gym import wrappers


class BaseAgent(object):
    def __init__(self, config: wandb.Config) -> None:
        env = gym.make(config.environment, visualize=True, step_mul=config.game_speed, random_seed=config.seed)
        self.env = wrappers.RecordEpisodeStatistics(env)
        self.env = wrappers.Monitor(
            env, directory="/tmp/betastar", force=True)
        self.episodes = config.episodes
    
    @property
    def sc2env(self):
        return self.game_env._env

    @property
    def game_env(self):
        return self.env.env
