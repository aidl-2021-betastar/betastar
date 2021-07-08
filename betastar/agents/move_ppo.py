from betastar.agents.ppo import PPO
from betastar.envs.env import PySC2Env
from betastar.models import ScreenNet
from torch import Tensor, nn


class MovePPO(PPO):
    def interpret_action(self, action: Tensor):
        return action[0] * self.config.screen_size + action[1]

    def get_model(self, env: PySC2Env) -> nn.Module:
        return ScreenNet(
            env.feature_original_shapes[0][0], self.config.screen_size
        )
