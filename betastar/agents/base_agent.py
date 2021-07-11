from pathlib import Path

import wandb
from betastar.envs.multiproc import MultiProcEnv
from torch import nn
import torch


class BaseAgent(object):
    def __init__(self, config: wandb.Config) -> None:
        self.config = config

        self.env = MultiProcEnv(
            self.config.environment,
            self.config.screen_size,
            count=self.config.num_workers,
            output_path=self.config.output_path
        )
        self.env.start()

    def last_video(self, step: int) -> wandb.Video:
        videos = list(Path(f"{self.config.output_path}/gym").glob("*.mp4"))
        videos.sort()
        return wandb.Video(str(videos[-1]), f"step={step}")

    def last_replay(self, environment_name: str, step_n: int) -> wandb.Artifact:
        artifact = wandb.Artifact(
            name=f"{environment_name}.{wandb.run.id}.{step_n}",
            type="replay",
            metadata={
                "environment": environment_name,
                "step": step_n,
            },
        )
        replays = list(Path(f"{self.config.output_path}/replays").glob("*.SC2Replay"))
        replays.sort()
        artifact.add_file(str(replays[-1]))
        return artifact

    def last_model(self, model: nn.Module, environment_name: str, step_n: int) -> wandb.Artifact:
        artifact = wandb.Artifact(
            name=f"model-{environment_name}.{wandb.run.id}.{step_n}",
            type="model",
            metadata={
                "environment": environment_name,
                "step": step_n,
            },
        )
        path = Path(f"{self.config.output_path}/last_model.pth")
        torch.save({
            'parameters': model.state_dict(),
            'config': self.config.as_dict()
        }, path) # type: ignore
        
        artifact.add_file(str(path))
        return artifact
