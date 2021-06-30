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


class MoveEnv(PySC2Env):
    """
    A super simplified PySC2 environment that only lets you move with your whole army to any (x,y) in the screen.
    """
    def close(self):
        self._env.close()

    def __init__(
        self,
        **kwargs
    ) -> None:

        super().__init__(**kwargs)
        self.action_space = spaces.MultiDiscrete(
            [
                1, # 0 - noop, 1 - move
                self.spatial_dim,
                self.spatial_dim
            ]
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, Done, Info]:
        what = action[0].item()
        if what == 0:
            act = [actions.FunctionCall(0, [])]
        else:
            act = [
                # select all army
                actions.FunctionCall(7, [0]),
                # and move to (action[1], action[2])
                actions.FunctionCall(331, [0, (action[1].item(), action[2].item())])
            ]

        response = self._env.step(act)[0]
        return (
            self._format_observation(response.observation),
            response.reward,
            response.step_type == StepType.LAST,
            {},
        )

    def reset_action_mask(self):
        """
        In this simplified environment, all actions are always available.
        """
        self.action_mask = np.ones(self.action_space.nvec.sum())  # type: ignore

    def _format_observation(self, raw_obs) -> Observation:
        obs = super()._format_observation(raw_obs)
        self.reset_action_mask()
        return obs

def spawn_env(environment: str, game_speed: int, spatial_dim: int, rank: int, monitor=False) -> PySC2Env:
    env = gym.make(environment, spatial_dim=spatial_dim, rank=rank, step_mul=game_speed, monitor=monitor)
    if monitor:
        env = wrappers.Monitor(env, directory="/tmp/betastar", force=True)
    return env  # type: ignore
