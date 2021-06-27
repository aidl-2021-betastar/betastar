from pathlib import Path
from typing import List, Tuple

import torch as T
import wandb
from betastar.envs.multiproc import MultiProcEnv
from betastar.data import Episode
from betastar.player import Player
from tqdm import tqdm


class BaseAgent(object):
    def __init__(self, config: wandb.Config) -> None:
        self.config = config

        self.env = MultiProcEnv(
            self.config.environment,
            self.config.game_speed,
            count=self.config.num_workers,
        )
        self.env.start()

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

    def play(self, player: Player) -> Tuple[List[Episode], List[float]]:
        episodes: List[Episode] = []
        episode_rewards: List[float] = []

        screen, minimap, non_spatial, _reward, _done, action_mask = self.env.reset()

        episodes = []
        episodes_ = [Episode() for n in range(self.config.num_workers)]
        episode_rewards = []
        episode_rewards_ = [0 for n in range(self.config.num_workers)]
        played = 0
        with tqdm(range(self.config.episodes), unit="episodes", leave=False) as pbar:
            while played < self.config.episodes:
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
                    episodes_[i].add(
                        (screen[i], minimap[i], non_spatial[i]),
                        actions[i],
                        rewards[i].item(),  # type: ignore
                        values[i].item(),  # type: ignore
                        next_values[i].item(),  # type: ignore
                        used_action_mask[i],
                        dones[i].item(),  # type: ignore
                    )
                    episode_rewards_[i] += rewards[i].item()  # type: ignore
                    if dones[i]:
                        episodes.append(episodes_[i])
                        episode_rewards.append(episode_rewards_[i])
                        episodes_[i] = Episode()
                        episode_rewards_[i] = 0

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

                    played += len(done_workers)
                    pbar.update(played)

        return episodes, episode_rewards
