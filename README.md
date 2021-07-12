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

In Reinforcement Learning  (RL) an agent interacts with a given environment via state visiting, action selection. After every pair (S,a) of these the environment returns a numerical reward (R) and then the agent transitions the next state (S'). Note that this transition may be entirely dependent on the 4 dimension vector (S,a,R,S') yielding a unique Markov Chain for each of the run episodes.

Developing and setting up the environment is a crucial yet not trivial part of any RL project. In this section, we will not provide this sort information; nonetheless, one can turn to section "PySC2 for Learning StarCraft2" for more details. Instead, here we will briefly review the Proximal Policy Optimization (PPO) algorithm which we chose to adapt the network parameters. Details on the network may be found in the following section. For the sake of clarity, we will not rely on the exact mathematical details but instead a 
provide a heuristic vision of such algorithm. However, appropiate references will always be provided.

In RL (Sutton & Barto, 2018; Maxim Lapan, 2020), actions are selected with a given probability. Note that in each one of the environment's state (S) not all the actions may be available. For this reason, we shall define a policy as the probability of selecting an action (a) at given state (S). The goal of any RL algorithm is to addequately modify this policy in order to select the most optimal actions in each state of the environment, hence the name of Policy Optimization. 

In the PPO implementation (John Schulman et al, 2017), a reasonable quantity to backpropagate inside the network is the ratio between the actual policy and the previous one. This ratio, which is entirely dependent on the values for each one of the network parameters, provides an estimate of the change that the policy has undergone after a given update. However, this proxy does not take into account whether the proposed change in the policy was either good or bad. For this reason, we need to weight this ratio by the "Advantatge" of having changed the policy with respect to the provious one. 

As the name suggests, the Advantatge function contains the total reward obtained when the agent has followed the new policy. One must note that by just computing the rewards a perfectly valid proxy for evaluating the goodness of the change is easily available. AQUÍ COMENTAR SOBRE PQ ES MILLOR AMB UNA BASELINE O CRITIC. SUAMENT TRANSICIONAR CAP A UNA GENERALITZACIÓ DE N-STEP I GAE. NO SERÀ FÀCIL...

## Neural Network

## PySC2 for Learning StarCraft2
