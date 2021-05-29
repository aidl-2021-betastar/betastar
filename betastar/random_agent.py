import gym
import wandb
from gym import wrappers
from tqdm import tqdm
import betastar.envs
import random
from absl import flags
FLAGS = flags.FLAGS
FLAGS([__file__])

def run(config: wandb.Config):
    env = gym.make(config.environment)
    env.settings['visualize'] = True
    env.settings['step_mul'] = config.game_speed
    env.settings['random_seed'] = random.seed(0)
    env = wrappers.Monitor(env, directory="/tmp/betastar-random", force=True)

    with tqdm(range(config.episodes), unit="episodes") as pbar:
        for episode in pbar:
            observation = env.reset()
            done = False
            t = 0
            while not done:
                action = env.action_space.sample()
                observation, reward, done, info = env.step(action)
                pbar.set_postfix(steps=str(t))
                t += 1

    env.close()
