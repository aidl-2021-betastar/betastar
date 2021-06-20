import os
import random

import click
import numpy as np
import torch
import torch.multiprocessing as mp
import wandb

from betastar.agents.a3c import A2C
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
    type=click.Choice(
        [
            "SC2Game-v0",
            "SC2MoveToBeacon-v0",
            "SC2MoveToBeacon-v1",
            "SC2CollectMineralShards-v0",
            "SC2CollectMineralShards-v1",
            "SC2CollectMineralShards-v2",
            "SC2FindAndDefeatZerglings-v0",
            "SC2DefeatRoaches-v0",
            "SC2DefeatZerglingsAndBanelings-v0",
            "SC2CollectMineralsAndGas-v0",
            "SC2BuildMarines-v0",
        ]
    ),
    default="SC2MoveToBeacon-v0",
)
@click.option(
    "--render-interval", default=200, help="How many episodes to skip between renders"
)
@click.option("--reward-decay", default=0.95, help="Gamma hyperparameter")
@click.option("--batch-size", default=32)
@click.option("--traj-length", default=16)
@click.option("--gae-lambda", default=0.95)
@click.option("--use-gae/--no-use-gae", default=True)
@click.option("--normalize-advantages/--no-normalize-advantages", default=True)
@click.option("--normalize-returns/--no-normalize-returns", default=True)
@click.option("--entropy-coeff", default=0.01, help="Entropy regularisation term")
@click.option("--critic-coeff", default=0.5, help="Critic regularisation term")
@click.option("--learning-rate", default=1e-4)
@click.option("--num-workers", default=int(mp.cpu_count()))
@click.option("--seed", default=42)
@click.option(
    "--epochs",
    default=5,
    help="Total number of [gather experiences / learn from experiences] cycles",
)
@click.option(
    "--game-speed",
    default=None,
    help="How many game steps per agent step (action/observation). None means use the map default",
)
@click.option(
    "--dryrun", is_flag=True, help="Whether to run wandb in dryrun mode or not"
)
def run(
    agent: str,
    environment: str,
    render_interval: int,
    reward_decay: float,
    batch_size: int,
    traj_length: int,
    gae_lambda: float,
    use_gae: bool,
    entropy_coeff: float,
    critic_coeff: float,
    learning_rate: float,
    num_workers: int,
    seed: int,
    epochs: int,
    dryrun: bool,
    game_speed: int,
    normalize_advantages: bool,
    normalize_returns: bool,
):
    if dryrun or agent == "random":
        os.environ["WANDB_MODE"] = "dryrun"

    wandb.init(
        project="betastar",
        entity="aidl-2021-betastar",
        config={
            "agent": agent,
            "render_interval": render_interval,
            "reward_decay": reward_decay,
            "batch_size": batch_size,
            "traj_length": traj_length,
            "gae_lambda": gae_lambda,
            "use_gae": use_gae,
            "entropy_coeff": entropy_coeff,
            "critic_coeff": critic_coeff,
            "learning_rate": learning_rate,
            "num_workers": num_workers,
            "seed": seed,
            "epochs": epochs,
            "environment": environment,
            "game_speed": game_speed,
            "normalize_advantages": normalize_advantages,
            "normalize_returns": normalize_returns,
        },
        monitor_gym=False,
    )

    config = wandb.config

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if agent == "random":
        RandomAgent(config).run()
    elif agent == "a3c":
        A2C(config).run()
