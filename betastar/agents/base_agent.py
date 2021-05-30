import gym
import random
import wandb
from gym import wrappers


class BaseAgent(object):
    def __init__(self, config: wandb.Config) -> None:
        env = gym.make(config.environment)
        env.settings['visualize'] = True
        env.settings['step_mul'] = config.game_speed
        env.settings['random_seed'] = random.seed(0)
        self.env = wrappers.Monitor(
            env, directory="/tmp/betastar-random", force=True)
        self.episodes = config.episodes
