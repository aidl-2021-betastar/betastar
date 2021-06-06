from betastar.envs.env import spawn_env
import numpy
import wandb
from absl import flags
from betastar.agents import base_agent
from tqdm import tqdm

FLAGS = flags.FLAGS
FLAGS([__file__])

class RandomAgent(base_agent.BaseAgent):
    def __init__(self, config: wandb.Config) -> None:
        super().__init__(config)

    def run(self):
        env = spawn_env(self.config.environment, self.config.game_speed)
        with tqdm(range(self.config.episodes_per_epoch), unit="episodes") as pbar:
            for episode in pbar:
                observation = env.reset()
                done = False
                t = 0
                while not done:
                    env.render()
                    action = self.env.random_action() # type: ignore
                    observation, reward, done, info = env.step(action)
                    pbar.set_postfix(steps=str(t))
                    t += 1

        env.close()