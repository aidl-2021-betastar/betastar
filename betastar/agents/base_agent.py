from betastar.envs.env import PySC2Env
import gym
import wandb
import betastar.envs

class BaseAgent(object):
    def __init__(self, config: wandb.Config) -> None:
        self.config = config