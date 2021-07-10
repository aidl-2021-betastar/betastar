from pathlib import Path

import wandb
from betastar.envs.multiproc import MultiProcEnv


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

    def last_video(self, step: int) -> wandb.Video:
        videos = list(Path(f"/tmp/betastar/{self.config.environment}").glob("*.mp4"))
        videos.sort()
        return wandb.Video(str(videos[-1]), f"step={step}")

    def last_replay(self, environment_name: str, step_n: int) -> wandb.Artifact:
        artifact = wandb.Artifact(
            name=f"{environment_name}.{step_n}",
            type="replay",
            metadata={
                "environment": environment_name,
                "step": step_n,
            },
        )
        replays = list(Path(f"/tmp/betastar/{self.config.environment}").glob("*.SC2Replay"))
        replays.sort()
        artifact.add_file(str(replays[-1]))
        return artifact
