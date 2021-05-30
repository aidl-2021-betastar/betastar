import gym
import wandb
from gym import wrappers
from tqdm import tqdm
import betastar.envs
import random
import numpy
from pysc2.lib import actions
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
                if env.env.__class__ == betastar.envs.SC2GameEnv:
                    action = select_action(observation, env)
                else:
                    action = env.action_space.sample()
                observation, reward, done, info = env.step(action)
                pbar.set_postfix(steps=str(t))
                t += 1

    env.close()

def select_action(obs, env):
    function_id = numpy.random.choice(obs.observation.available_actions)
    args = [[numpy.random.randint(0, size) for size in arg.sizes]
            for arg in env.env._env.action_spec()[0].functions[function_id].args]
    return [function_id, *args]
