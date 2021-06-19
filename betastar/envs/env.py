# from: https://raw.githubusercontent.com/vwxyzjn/gym-pysc2/master/gym_pysc2/envs/pysc2env.py
# which was inspired by: https://github.com/inoryy/reaver/blob/master/reaver/envs/sc2.py
import time
from typing import List, Tuple
from click import Path

import gym
import numpy as np
import pygame
import torch as T
from gym import spaces

from pysc2.env import sc2_env
from pysc2.env.environment import StepType
from pysc2.lib import protocol
from pysc2.lib import actions, features


from gym import wrappers
from torch import Tensor
import wandb

Info = dict
Observation = Tuple[Tensor, Tensor, Tensor]
Reward = float
Value = Reward
Done = bool
Action = np.ndarray
ActionMask = Tensor


class PySC2Env(gym.Env):
    metadata = {"render.modes": ["rgb_array", "human"]}

    action_ids: List[int]

    def close(self):
        self._env.close()

    def __init__(
        self,
        action_ids: List[int] = [],
        spatial_dim=64,
        step_mul=8,
        map_name="MoveToBeacon",
        visualize=True,
    ) -> None:

        super().__init__()
        self.action_ids = action_ids
        self.spatial_dim = spatial_dim
        self.step_mul = step_mul
        self.map_name = map_name
        self.visualize = visualize

        # preprocess
        if len(self.action_ids) == 0:
            self.action_ids = [f.id for f in actions.FUNCTIONS]  # type: ignore
        self.reverse_action_ids = np.zeros(max(self.action_ids) + 1, dtype=np.int16)
        for idx, aid in enumerate(self.action_ids):
            self.reverse_action_ids[aid] = idx
        
        self._env = sc2_env.SC2Env(
            map_name=self.map_name,
            visualize=self.visualize,
            agent_interface_format=[
                sc2_env.parse_agent_interface_format(
                    feature_screen=self.spatial_dim,
                    feature_minimap=self.spatial_dim,
                    rgb_screen=None,
                    rgb_minimap=None,
                )
            ],
            save_replay_episodes=1,
            replay_dir='/tmp/betastar',
            step_mul=self.step_mul,
            players=[sc2_env.Agent(sc2_env.Race.terran)],
        )
        self.temp_obs = self._env.reset()[0].observation

        # observation space
        obs_features = {
            "screen": [
                "player_relative",
                "selected",
                "visibility_map",
                "unit_hit_points_ratio",
                "unit_density",
            ],
            "minimap": ["player_relative", "selected", "visibility_map", "camera"],
            "non-spatial": ["available_actions", "player"],
        }
        screen_feature_to_idx = {
            feat: idx for idx, feat in enumerate(features.SCREEN_FEATURES._fields)
        }
        minimap_feature_to_idx = {
            feat: idx for idx, feat in enumerate(features.MINIMAP_FEATURES._fields)
        }
        self.feature_masks = {
            "screen": [screen_feature_to_idx[f] for f in obs_features["screen"]],
            "minimap": [minimap_feature_to_idx[f] for f in obs_features["minimap"]],
        }

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                len(obs_features["screen"]) * spatial_dim * spatial_dim
                + len(obs_features["minimap"]) * spatial_dim * spatial_dim
                + len(self.action_ids)
                + len(self.temp_obs["player"]),
            ),
        )
        self.feature_flatten_shapes = (
            (len(obs_features["screen"]) * spatial_dim * spatial_dim,)
            + (len(obs_features["minimap"]) * spatial_dim * spatial_dim,)
            + (len(self.action_ids) + len(self.temp_obs["player"]),)
        )
        self.feature_original_shapes = [
            (len(obs_features["screen"]), spatial_dim, spatial_dim),
            (len(obs_features["minimap"]), spatial_dim, spatial_dim),
            (len(self.action_ids) + len(self.temp_obs["player"]),),
        ]

        # action space
        self.args = [
            "screen",
            "minimap",
            "screen2",
            "queued",
            "control_group_act",
            "control_group_id",
            "select_add",
            "select_point_act",
            "select_unit_act",
            # 'select_unit_id'
            "select_worker",
            "build_queue_id",
            # 'unload_id'
        ]

        self.args_idx = {}
        action_args = ()
        for arg_name in self.args:
            arg = getattr(self._env.action_spec()[0][0], arg_name)
            self.args_idx[arg_name] = slice(
                len(action_args) + 1, len(action_args) + 1 + len(arg.sizes)
            )
            action_args += arg.sizes
        self.action_space = spaces.MultiDiscrete(
            [
                len(self.action_ids),
            ]
            + list(action_args)
        )

    def random_action(self) -> Action:
        action = self.action_space.sample()
        if self.is_available_action(action[0]):
            return action
        else:
            return np.zeros_like(action)  # no-op

    def is_available_action(self, action_idx: int) -> bool:
        return self.action_mask[np.arange(self.action_space.nvec[0])][action_idx] == 1.0  # type: ignore

    def step(self, action) -> Tuple[Observation, Reward, Done, Info]:
        defaults = {
            "control_group_act": 0,
            "control_group_id": 0,
            "select_point_act": 0,
            "select_unit_act": 0,
            "select_unit_id": 0,
            "build_queue_id": 0,
            "unload_id": 0,
        }
        action_id_idx, args = action[0], []
        action_id = self.action_ids[action_id_idx]
        for arg_type in actions.FUNCTIONS[action_id].args:  # type: ignore
            arg_name = arg_type.name
            if arg_name in self.args:
                arg = action[self.args_idx[arg_name]]
                args.append(arg)
            else:
                args.append([defaults[arg_name]])

        response = self._env.step([actions.FunctionCall(action_id, args)])[0]
        return (
            self._format_observation(response.observation),
            response.reward,
            response.step_type == StepType.LAST,
            {},
        )

    def reset(self) -> Observation:
        response = self._env.reset()[0]
        return self._format_observation(response.observation)

    def _format_observation(self, raw_obs) -> Observation:
        # action masking
        action_id_mask = np.zeros(len(self.action_ids))
        for available_action_id in raw_obs["available_actions"]:
            if available_action_id in self.reverse_action_ids:
                action_id_mask[self.reverse_action_ids[available_action_id]] = 1
        self.action_mask = np.ones(self.action_space.nvec.sum())  # type: ignore
        self.action_mask[: len(self.action_ids)] = action_id_mask
        self.available_actions = raw_obs["available_actions"]

        screen = T.from_numpy(
            raw_obs["feature_screen"][self.feature_masks["screen"]].flatten()
        ).float()
        minimap = T.from_numpy(
            raw_obs["feature_minimap"][self.feature_masks["minimap"]].flatten()
        ).float()

        available_actions = np.zeros(len(self.action_ids))
        for available_action_id in raw_obs["available_actions"]:
            if available_action_id in self.reverse_action_ids:
                available_actions[self.reverse_action_ids[available_action_id]] = 1

        non_spatial = T.from_numpy(
            np.concatenate([available_actions, raw_obs["player"]])
        ).float()

        screen_shape, minimap_shape, non_spatial_shape = self.feature_original_shapes

        return (
            screen.reshape(screen_shape),
            minimap.reshape(minimap_shape),
            non_spatial.reshape(non_spatial_shape),
        )

    def render(self, mode="human"):
        if mode == "rgb_array" and self._env._renderer_human._window is not None:
            x = self._env._renderer_human._window.copy()
            array = pygame.surfarray.pixels3d(x)
            array = np.transpose(array, axes=(1, 0, 2))
            del x
            return array


def spawn_env(environment: str, game_speed: int, monitor=False) -> PySC2Env:
    env = None
    while True:
        try:
            env = gym.make(environment, visualize=monitor, step_mul=game_speed)
            if monitor:
                env = wrappers.RecordEpisodeStatistics(env)
                env = wrappers.Monitor(env, directory="/tmp/betastar", resume=True)
            return env  # type: ignore
        except protocol.ConnectionError:
            if env is not None:
                env.close()
            print("SHITTTTTTTTTTTT trying one more time")
            time.sleep(1) # deep breath