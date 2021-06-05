import gym
import wandb
from gym import wrappers
import betastar.envs


class BaseAgent(object):
    def __init__(self, config: wandb.Config) -> None:
        env = gym.make(config.environment, visualize=True, step_mul=config.game_speed)
        self.env = wrappers.RecordEpisodeStatistics(env)
        self.env = wrappers.Monitor(
            env, directory="/tmp/betastar", force=True)
        self.episodes = config.episodes
