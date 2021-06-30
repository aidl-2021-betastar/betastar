# from: https://raw.githubusercontent.com/vwxyzjn/gym-pysc2/master/gym_pysc2/envs/pysc2env.py
# which was inspired by: https://github.com/inoryy/reaver/blob/master/reaver/envs/sc2.py
from typing import List, Tuple

import gym
import numpy as np
import pygame
import torch as T
from gym import spaces, wrappers
from pysc2.env import sc2_env
from pysc2.env.environment import StepType
from pysc2.lib import actions, features
from torch import Tensor
from .env import PySC2Env

Info = dict
Observation = Tuple[Tensor, Tensor, Tensor]
Reward = float
Value = Reward
Done = bool
Action = Tensor
ActionMask = Tensor

_NO_OP = actions.FUNCTIONS.no_op.id

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_NOT_QUEUED = [0]


class MoveEnv(PySC2Env):
    """
    A super simplified PySC2 environment that only lets you move with your whole army to any (x,y) in the screen.
    """
    def close(self):
        self._env.close()

    def reset(self):
        response = self._env.reset()[0]
        response = self._env.step([actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])])[0]
        return self._format_observation(response.observation)

    def __init__(
        self,
        **kwargs
    ) -> None:

        super().__init__(**kwargs)
        self.action_space = spaces.Discrete(
            self.spatial_dim * self.spatial_dim
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, Done, Info]:
        y = action.item() // self.spatial_dim
        x = action.item() % self.spatial_dim

        response = self._env.step([actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [y, x]])])[0]
        return (
            self._format_observation(response.observation),
            response.reward,
            response.step_type == StepType.LAST,
            {},
        )

def spawn_env(environment: str, game_speed: int, spatial_dim: int, rank: int, monitor=False) -> PySC2Env:
    env = gym.make(environment, spatial_dim=spatial_dim, rank=rank, step_mul=game_speed, monitor=monitor)
    if monitor:
        env = wrappers.Monitor(env, directory="/tmp/betastar", force=True)
    return env  # type: ignore
