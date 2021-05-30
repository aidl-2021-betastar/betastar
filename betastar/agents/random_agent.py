import numpy
import wandb
from absl import flags
from betastar.agents import base_agent
from betastar.envs import SC2GameEnv
from tqdm import tqdm

FLAGS = flags.FLAGS
FLAGS([__file__])


class RandomAgent(base_agent.BaseAgent):
    def __init__(self, config: wandb.Config) -> None:
        super().__init__(config)

    def run(self):
        with tqdm(range(self.episodes), unit="episodes") as pbar:
            for episode in pbar:
                observation = self.env.reset()
                done = False
                t = 0
                while not done:
                    self.env.render()
                    if self.game_env.__class__ == SC2GameEnv:
                        action = self.select_action()
                    else:
                        action = self.env.action_space.sample()
                    observation, reward, done, info = self.env.step(action)
                    pbar.set_postfix(steps=str(t))
                    t += 1

        self.env.close()

    def select_action(self):
        function_id = numpy.random.choice(self.game_env.available_actions)
        args = [[numpy.random.randint(0, size) for size in arg.sizes]
                for arg in self.sc2env.action_spec()[0].functions[function_id].args]
        return [function_id, *args]
