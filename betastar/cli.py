from betastar.agents.a3c import A3C
import os
import random

import click
import numpy as np
import torch
import torch.multiprocessing as mp
import wandb

from betastar.agents.random_agent import RandomAgent


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--agent",
    "-a",
    type=click.Choice(["random", "a3c"]),
    default="random",
)
@click.option(
    "--environment",
    "-e",
    type=click.Choice([
        'SC2Game-v0', 'SC2MoveToBeacon-v0', 'SC2MoveToBeacon-v1', 'SC2CollectMineralShards-v0', 'SC2CollectMineralShards-v1', 'SC2CollectMineralShards-v2', 'SC2FindAndDefeatZerglings-v0', 'SC2DefeatRoaches-v0', 'SC2DefeatZerglingsAndBanelings-v0', 'SC2CollectMineralsAndGas-v0', 'SC2BuildMarines-v0',
    ]),
    default="SC2MoveToBeacon-v0",
)
@click.option("--episodes", default=5, help="Number of episodes to run.")
@click.option(
    "--render-interval", default=200, help="How many episodes to skip between renders"
)
@click.option("--reward-decay", default=0.95, help="Gamma hyperparameter")
@click.option("--learning-rate", default=1e-2)
@click.option("--num-workers", default=int(mp.cpu_count() / 2.0))
@click.option("--seed", default=42)
@click.option("--game-speed", default=None, help="How many game steps per agent step (action/observation). None means use the map default")
@click.option("--dryrun", is_flag=True, help="Whether to run wandb in dryrun mode or not")
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
    game_speed: int,
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
            "environment": environment,
            "game_speed": game_speed,
        },
        monitor_gym=True,
    )

    config = wandb.config

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if agent == "random":
        RandomAgent(config).run()
    elif agent == "a3c":
        A3C(config).run()
