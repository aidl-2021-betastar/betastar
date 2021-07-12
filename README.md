# betastar

Mastering StarCraft 2 minigames with Reinforcement Learning.

## Training an agent

This will train a PPO agent for 1000 game steps (10M is more of a real training session) on MoveToBeacon and save replays, videos and the model in `./out`:

```
docker build aidl/betastar:latest .
docker run -it -v $PWD/out:/output aidl/betastar:latest train --agent full_ppo --environment SC2MoveToBeacon-v0 --output-path /output --total-steps 1000
```

When wandb asks you for an API key (that's why we need the `-it`), you can enter one or select the third option (Don't visualize my results).

## Playing with a trained agent

From the previous step, you'll have a model at `./out/RUN_ID/last_model.pth`. You can see it in action like this:

```
docker build aidl/betastar:latest .
docker run -it -v $PWD/out:/output aidl/betastar:latest play --model /output/RUN_ID/last_model.pth --episodes 4 --output-path /output
```

Replays and videos from this playtest will be under `./out` as well.

## Setup for development

1. Make sure you have Docker installed on your machine.
2. Install VSCode.
3. When you open the folder, it'll ask you to "Reopen in Container". Do it, and you're set!
4. You might need to re-run `pip install poetry` after starting the container.

## Reinforcement Learning Algorithm

In Reinforcement Learning  (RL) an agent interacts with a given environment via state visiting, action selection. After every pair $(S,a)$ of these the environment returns a numerical reward $ (R) $ and then the agent transitions the next state $(S')$. Note that this transition may be entirely dependent on the 4 dimension vector $(S,a,R,S')$ yielding a unique Markov Chain for each of the run episodes.
