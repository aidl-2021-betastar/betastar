import os
import random

import click
import numpy as np
import torch
import torch.multiprocessing as mp
import wandb
from pyvirtualdisplay import Display

from betastar.agents.spatial_a2c import SpatialA2C
from betastar.agents.random_agent import RandomAgent


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--agent",
    "-a",
    type=click.Choice(["random", "spatial_a2c"]),
    default="random",
)
@click.option(
    "--environment",
    "-e",
    type=click.Choice(
        [
            "SC2Game-v0",
            "SC2MoveToBeacon-v0",
            "SC2MoveToBeaconSimple-v0",
            "SC2MoveToBeaconSimpleOrNot-v0",
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
@click.option("--episodes", default=4)
@click.option("--total-steps", default=1000000)
@click.option("--batch-size", default=32)
@click.option("--unroll-length", default=60)
@click.option("--gae-lambda", default=0.95)
@click.option("--use-gae/--no-use-gae", default=True)
@click.option("--use-ppo/--no-use-ppo", default=False)
@click.option("--clip-range", default=0.1)
@click.option("--normalize-advantages/--no-normalize-advantages", default=True)
@click.option("--normalize-returns/--no-normalize-returns", default=True)
@click.option("--anneal-lr/--no-anneal-lr", default=False)
@click.option("--entropy-coeff", default=0.01, help="Entropy regularisation term")
@click.option("--critic-coeff", default=0.5, help="Critic regularisation term")
@click.option("--learning-rate", default=2.5e-4)
@click.option("--max-grad-norm", default=0.5)
@click.option(
    "--test-interval",
    default=100,
    help="Play/learn cycles to wait before running a test round",
)
@click.option("--num-workers", default=int(mp.cpu_count()))
@click.option("--seed", default=42)
@click.option(
    "--update-epochs",
    default=4,
    help="Number of times to learn from each experience",
)
@click.option(
    "--screen-size",
    default=16,
    help="Play in a ?x? screen",
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
    episodes: int,
    total_steps: int,
    batch_size: int,
    unroll_length: int,
    gae_lambda: float,
    use_gae: bool,
    use_ppo: bool,
    clip_range: float,
    entropy_coeff: float,
    critic_coeff: float,
    learning_rate: float,
    num_workers: int,
    test_interval: int,
    seed: int,
    dryrun: bool,
    game_speed: int,
    update_epochs: int,
    max_grad_norm: float,
    screen_size: int,
    normalize_advantages: bool,
    normalize_returns: bool,
    anneal_lr: bool
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
            "episodes": episodes,
            "update_epochs": update_epochs,
            "total_steps": total_steps,
            "batch_size": batch_size,
            "unroll_length": unroll_length,
            "gae_lambda": gae_lambda,
            "use_gae": use_gae,
            "use_ppo": use_ppo,
            "clip_range": clip_range,
            "entropy_coeff": entropy_coeff,
            "critic_coeff": critic_coeff,
            "learning_rate": learning_rate,
            "num_workers": num_workers,
            "seed": seed,
            "environment": environment,
            "game_speed": game_speed,
            "test_interval": test_interval,
            "max_grad_norm": max_grad_norm,
            "screen_size": screen_size,
            "normalize_advantages": normalize_advantages,
            "normalize_returns": normalize_returns,
            "anneal_lr": anneal_lr
        },
        monitor_gym=False,
    )

    config = wandb.config

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    display = Display(visible=False, size=(800, 600))
    display.start()

    if agent == "random":
        RandomAgent(config).run()
    elif agent == "spatial_a2c":
        SpatialA2C(config).run()

    display.stop()
    
