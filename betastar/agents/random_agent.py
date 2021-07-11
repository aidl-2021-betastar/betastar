import wandb
from absl import flags
from betastar.agents import base_agent
from betastar.envs.env import spawn_env
from tqdm import tqdm
import torch as T

FLAGS = flags.FLAGS
FLAGS([__file__])


class RandomAgent(base_agent.BaseAgent):
    def __init__(self, config: wandb.Config) -> None:
        super().__init__(config)

    def run(self):
        env = spawn_env(
            self.config.environment,
            spatial_dim=self.config.screen_size,
            output_path="./output",
        )
        with tqdm(range(4), unit="episodes") as pbar:
            for episode in pbar:
                observation = env.reset()
                done = False
                t = 0
                while not done:
                    env.render()
                    action = env.random_action()  # type: ignore
                    observation, reward, done, info = env.step(T.from_numpy(action))
                    pbar.set_postfix(steps=str(t))
                    t += 1

        env.close()
