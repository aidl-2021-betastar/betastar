import gym
import wandb
from gym import wrappers
from tqdm import tqdm
import betastar.env # import PySC2Env


def run(config: wandb.Config):
    env = wrappers.Monitor(gym.make(config.environment), directory="/tmp/betastar-random", force=True)

    with tqdm(range(config.episodes), unit="episodes") as pbar:
        for episode in pbar:
            observation = env.reset()
            for t in range(100):
                if episode % config.render_interval == 0:
                    env.render()
                action = env.action_space.sample()
                observation, reward, done, info = env.step(action)
                pbar.set_postfix(steps=str(t))
                if done:
                    break

    env.close()
