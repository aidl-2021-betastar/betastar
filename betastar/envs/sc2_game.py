"""
   Copyright 2017 Islam Elnabarawy

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
# Imported from https://github.com/islamelnabarawy/sc2gym

import logging
from pathlib import Path
import wandb
import gym
import pygame
import numpy as np
from pysc2.env import sc2_env
from pysc2.env.environment import StepType
from pysc2.lib import actions

__author__ = 'Islam Elnabarawy'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_NO_OP = actions.FUNCTIONS.no_op.id # type: ignore


class SC2GameEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array', 'human']}
    default_settings = {
        'agent_interface_format': sc2_env.parse_agent_interface_format(
            feature_screen=84,
            feature_minimap=64,
        ),
        'players': [sc2_env.Agent(sc2_env.Race.terran)],
        'save_replay_episodes': 1,
        'replay_dir': '/tmp/betastar'
    }

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._kwargs = kwargs
        self._env = None

        self._episode = 0
        self._num_step = 0
        self._episode_reward = 0
        self._total_reward = 0

    def step(self, action):
        return self._safe_step(action)

    def _safe_step(self, action):
        self._num_step += 1
        if action[0] not in self.available_actions:
            logger.warning("Attempted unavailable action: %s", action)
            action = [_NO_OP]
        try:
            obs = self._env.step(
                [actions.FunctionCall(action[0], action[1:])])[0]
        except KeyboardInterrupt:
            logger.info("Interrupted. Quitting...")
            return None, 0, True, {}
        except Exception:
            logger.exception(
                "An unexpected error occurred while applying action to environment.")
            return None, 0, True, {}
        self.available_actions = obs.observation['available_actions']
        reward = obs.reward
        self._episode_reward += reward
        self._total_reward += reward
        return obs, reward, obs.step_type == StepType.LAST, {}

    def last_replay(self) -> wandb.Artifact:
        artifact = wandb.Artifact(name=f"{self._env.map_name}.{self._episode}", type='replay', metadata={'map': self._env.map_name, 'reward': self._episode_reward, 'episode': self._episode})
        replays = list(Path('/tmp/betastar').glob('*.SC2Replay'))
        replays.sort()
        artifact.add_file(str(replays[-1]))
        return artifact

    def render(self, mode="human"):
        if mode == "rgb_array":
            x = self._env._renderer_human._window.copy()
            array = pygame.surfarray.pixels3d(x)
            array = np.transpose(array, axes=(1, 0, 2))
            del x
            return array

    def reset(self):
        if self._env is None:
            self._init_env()
        if self._episode > 0:
            logger.info("Episode %d ended with reward %d after %d steps.",
                        self._episode, self._episode_reward, self._num_step)
            logger.info("Got %d total reward so far, with an average reward of %g per episode",
                        self._total_reward, float(self._total_reward) / self._episode)
            wandb.log({
                'episode_reward/now': self._episode_reward,
                'episode_reward/avg': float(self._total_reward) / self._episode,
                }, step=self._episode)
            wandb.log_artifact(self.last_replay())
        self._episode += 1
        self._num_step = 0
        self._episode_reward = 0
        logger.info("Episode %d starting...", self._episode)
        obs = self._env.reset()[0]
        self.available_actions = obs.observation['available_actions']
        return obs

    def save_replay(self, replay_dir):
        self._env.save_replay(replay_dir)

    def _init_env(self):
        args = {**self.default_settings, **self._kwargs}
        logger.debug("Initializing SC2Env with settings: %s", args)
        print(f"Initializing SC2Env with settings: {args}")
        self._env = sc2_env.SC2Env(**args)

    def close(self):
        if self._episode > 0:
            logger.info("Episode %d ended with reward %d after %d steps.",
                        self._episode, self._episode_reward, self._num_step)
            logger.info("Got %d total reward, with an average reward of %g per episode",
                        self._total_reward, float(self._total_reward) / self._episode)
            wandb.log({'episode_reward/now': self._episode_reward, 'episode_reward/avg': float(self._total_reward) / self._episode}, step=self._episode)
        if self._env is not None:
            self._env.close()
        super().close()

    @property
    def settings(self):
        return self._kwargs
    
    @property
    def action_spec(self):
        if self._env is None:
            self._init_env()
        return self._env.action_spec()

    @property
    def observation_spec(self):
        if self._env is None:
            self._init_env()
        return self._env.observation_spec()

    @property
    def episode(self):
        return self._episode

    @property
    def num_step(self):
        return self._num_step

    @property
    def episode_reward(self):
        return self._episode_reward

    @property
    def total_reward(self):
        return self._total_reward
