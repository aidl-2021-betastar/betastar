import os
import random

import click
import numpy as np
import torch
import torch.multiprocessing as mp
import wandb

from betastar import random_agent


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--agent",
    "-a",
    type=click.Choice(["random"]),
    default="random",
)
@click.option(
    "--environment",
    "-e",
    type=click.Choice(["SC2MoveToBeacon-v0"]),
    default="SC2MoveToBeacon-v0",
)
@click.option("--episodes", default=500, help="Number of episodes to run.")
@click.option(
    "--render-interval", default=200, help="How many episodes to skip between renders"
)
@click.option("--reward-decay", default=0.95, help="Gamma hyperparameter")
@click.option("--learning-rate", default=1e-2)
@click.option("--num-workers", default=int(mp.cpu_count() / 2.0))
@click.option("--seed", default=42)
@click.option("--dryrun", is_flag=True)
def run(
    agent: str,
    environment: str,
    episodes: int,
    render_interval: int,
    reward_decay: float,
    learning_rate: float,
    num_workers: int,
    seed: int,
    dryrun: bool,
):
    if dryrun or agent == "random":
        os.environ["WANDB_MODE"] = "dryrun"

    wandb.init(
        project="betastar",
        entity="aidl-2021-betastar",
        config={
            "agent": agent,
            "episodes": episodes,
            "render_interval": render_interval,
            "reward_decay": reward_decay,
            "learning_rate": learning_rate,
            "num_workers": num_workers,
            "seed": seed,
            "environment": environment
        },
        monitor_gym=True,
    )

    config = wandb.config

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if agent == "random":
        random_agent.run(config)