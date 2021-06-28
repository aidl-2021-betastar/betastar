from pathlib import Path
from typing import List, Tuple

import torch as T
import wandb
from betastar.envs.multiproc import MultiProcEnv
from betastar.data import Trajectory
from betastar.player import Player
from tqdm import tqdm


class BaseAgent(object):
    def __init__(self, config: wandb.Config) -> None:
        self.config = config

        self.env = MultiProcEnv(
            self.config.environment,
            self.config.game_speed,
            self.config.screen_size,
            count=self.config.num_workers,
        )
        self.env.start()

        self.test_env = MultiProcEnv(
            self.config.environment,
            self.config.game_speed,
            self.config.screen_size,
            count=1
        )
        self.test_env.start()

    def last_video(self, step: int) -> wandb.Video:
        videos = list(Path("/tmp/betastar").glob("*.mp4"))
        videos.sort()
        return wandb.Video(str(videos[-1]), f"step={step}")

    def last_replay(self, map_name: str, step_n: int) -> wandb.Artifact:
        artifact = wandb.Artifact(
            name=f"{map_name}.{step_n}",
            type="replay",
            metadata={
                "map": map_name,
                "step": step_n,
            },
        )
        replays = list(Path("/tmp/betastar").glob("*.SC2Replay"))
        replays.sort()
        artifact.add_file(str(replays[-1]))
        return artifact

    def test(self, player: Player, episodes = 5) -> List[float]:
        rewards: List[float] = []
        for episode in tqdm(range(episodes), unit="episodes"):
            screen, minimap, non_spatial, _reward, _done, action_mask = self.test_env.reset()

            done = False
            ep_reward = 0.0

            while not done:
                with T.no_grad():
                    actions = player.act(screen, minimap, non_spatial, action_mask)
                    values = player.evaluate(screen, minimap, non_spatial)

                (
                    screen,
                    minimap,
                    non_spatial,
                    rewards_,
                    dones,
                    action_mask,
                ) = self.test_env.step(actions)

                ep_reward += rewards_[0].item()

                done = dones[0]

            rewards.append(ep_reward)

        return rewards


    def play(self, player: Player, reset=False) -> Tuple[List[Trajectory], List[float]]:
        trajectories: List[Trajectory] = []
        trajectory_rewards: List[float] = []

        if reset:
            screen, minimap, non_spatial, _reward, _done, action_mask = self.env.reset()
        else:
            screen, minimap, non_spatial, _reward, _done, action_mask = self.env.last_observation

        trajectories = [Trajectory() for n in range(self.config.num_workers)]
        trajectory_rewards = [0 for n in range(self.config.num_workers)]
        for step in tqdm(range(self.config.unroll_length), unit="steps", leave=False):
            with T.no_grad():
                actions = player.act(screen, minimap, non_spatial, action_mask)
                used_action_mask = action_mask.clone()
                values = player.evaluate(screen, minimap, non_spatial)

            (
                screen,
                minimap,
                non_spatial,
                rewards,
                dones,
                action_mask,
            ) = self.env.step(actions)

            with T.no_grad():
                next_values = player.evaluate(screen, minimap, non_spatial)

            for i in range(self.config.num_workers):
                trajectories[i].add(
                    (screen[i], minimap[i], non_spatial[i]),
                    actions[i],
                    rewards[i].item(),  # type: ignore
                    values[i].item(),  # type: ignore
                    next_values[i].item(),  # type: ignore
                    used_action_mask[i],
                    dones[i].item(),  # type: ignore
                )
                trajectory_rewards[i] += rewards[i].item()  # type: ignore

            # reset done workers and set their next state to the initial observations
            done_workers = T.where(dones == True)[0]
            if len(done_workers) > 0:
                (
                    screen_,
                    minimap_,
                    non_spatial_,
                    _reward,
                    _done,
                    action_mask_,
                ) = self.env.reset(only=done_workers.tolist())
                screen[done_workers] = screen_
                minimap[done_workers] = minimap_
                non_spatial[done_workers] = non_spatial_
                action_mask[done_workers] = action_mask_

        return trajectories, trajectory_rewards
