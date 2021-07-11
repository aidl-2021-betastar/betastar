import os
import random
from pathlib import Path

import click
import numpy as np
import torch
import torch.multiprocessing as mp
import wandb
from pyvirtualdisplay import Display

from betastar.agents import FullPPO, MovePPO, RandomAgent


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--agent",
    "-a",
    type=click.Choice(["random", "move_ppo", "full_ppo"]),
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
            "SC2MoveToBeaconMini-v0",
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
@click.option("--reward-decay", default=0.95, help="Gamma hyperparameter")
@click.option("--total-steps", default=1000000)
@click.option("--unroll-length", default=60)
@click.option("--trace-decay", default=0.95, help="Alpha hyperparameter (GAE)")
@click.option("--use-gae/--no-use-gae", default=False)
@click.option("--use-ppo/--no-use-ppo", default=False)
@click.option("--clip-range", default=0.1)
@click.option("--normalize-advantages/--no-normalize-advantages", default=False)
@click.option("--normalize-returns/--no-normalize-returns", default=False)
@click.option("--entropy-coeff", default=0.01, help="Entropy regularisation term")
@click.option("--critic-coeff", default=0.5, help="Critic regularisation term")
@click.option("--learning-rate", default=2.5e-4)
@click.option(
    "--checkpoint-interval",
    default=100,
    help="Play/learn cycles to wait before saving replay, video and model",
)
@click.option("--num-workers", default=int(mp.cpu_count()))
@click.option("--seed", default=42)
@click.option(
    "--screen-size",
    default=16,
    help="Play in a ?x? screen",
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(dir_okay=True, file_okay=False),
    default="./output",
)
@click.option(
    "--dryrun", is_flag=True, help="Whether to run wandb in dryrun mode or not"
)
def run(
    agent: str,
    environment: str,
    reward_decay: float,
    total_steps: int,
    unroll_length: int,
    trace_decay: float,
    use_gae: bool,
    use_ppo: bool,
    clip_range: float,
    entropy_coeff: float,
    critic_coeff: float,
    learning_rate: float,
    num_workers: int,
    checkpoint_interval: int,
    seed: int,
    dryrun: bool,
    screen_size: int,
    normalize_advantages: bool,
    normalize_returns: bool,
    output_path: str,
):
    if dryrun or agent == "random":
        os.environ["WANDB_MODE"] = "dryrun"

    wandb.init(
        project="betastar",
        entity="aidl-2021-betastar",
        config={
            "agent": agent,
            "reward_decay": reward_decay,
            "total_steps": total_steps,
            "unroll_length": unroll_length,
            "trace_decay": trace_decay,
            "use_gae": use_gae,
            "use_ppo": use_ppo,
            "clip_range": clip_range,
            "entropy_coeff": entropy_coeff,
            "critic_coeff": critic_coeff,
            "learning_rate": learning_rate,
            "num_workers": num_workers,
            "seed": seed,
            "environment": environment,
            "checkpoint_interval": checkpoint_interval,
            "screen_size": screen_size,
            "normalize_advantages": normalize_advantages,
            "normalize_returns": normalize_returns,
            "output_path": output_path,
        },
        monitor_gym=False,
    )

    config = wandb.config
    config.update(
        {"output_path": str((Path(config.output_path) / wandb.run.id).absolute())},
        allow_val_change=True,
    )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    display = Display(visible=False, size=(800, 600))
    display.start()

    if agent == "random":
        RandomAgent(config).run()
    elif agent == "move_ppo":
        MovePPO(config).run()
    elif agent == "full_ppo":
        FullPPO(config).run()

    display.stop()


@cli.command()
@click.option("--model", "-m", type=click.Path(exists=True, dir_okay=False))
@click.option("--episodes", "-e", type=int, default=4)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(dir_okay=True, file_okay=False),
    default="./output",
)
def play(
    model: str,
    episodes: int,
    output_path: str,
):
    display = Display(visible=False, size=(800, 600))
    display.start()

    os.environ["WANDB_MODE"] = "dryrun"

    saved = torch.load(model, map_location=torch.device("cpu"))
    wandb.init(config=saved["config"])  # type: ignore
    wandb.config.update(
        {
            "output_path": str((Path(output_path) / wandb.run.id).absolute()),
            "num_workers": 1,
            "episodes": episodes,
        },
        allow_val_change=True,
    )

    FullPPO(wandb.config).play(saved["parameters"])  # type: ignore

    display.stop()
