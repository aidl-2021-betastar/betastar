from typing import List
from betastar.agents.ppo import PPO
from betastar.envs.env import PySC2Env
from betastar.models import FullNet
import torch as T
from torch import Tensor, nn


class FullPPO(PPO):
    def interpret_action(self, action: List[Tensor]) -> Tensor:
        return T.stack(action).T

    def get_model(self, env: PySC2Env) -> nn.Module:
        return FullNet(env)
