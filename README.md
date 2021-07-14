# Betastar

Mastering StarCraft 2 minigames with Reinforcement Learning.
Authors: Txus Bach, Luis M. Etxenike, Joan Falcó, Oriol Gual
Advisor: Dani Fojo

## Introduction

Reinforcement Learning (RL) has been studied for more than 4 decades, however, it became famous right after DeepMind released a paper where they trained deep artificial neural networks to play a large amount of Atari Games achieving super-human scores. Since then, more sofisticated games have also been solved including Chess and Go. In a recent paper by a team in DeepMind (Vinyals et al, 2017) layed the ground for tackling a much more complex problem: building a RL model that could play StarCraft 2. In this project we attempt to solve some minigames of this game. 

This report is organized as follows. Initially, the description of the environment can be found. Next, we provide a brief sumamry of what is Reinforcement Learning as well as the algorithms we implemented. Then, we shall give details on the networks we trained as well as their limitations. Finally, the instructions to install all the dependencies as well as the general conclusions are presented.

## PySC2: StarCraft 2 environment for python

![1_-l5zyiih9yvox7o_0W_S2Q](https://user-images.githubusercontent.com/75299844/125631224-7bbbe707-4558-4092-8d9d-26056ee461cc.png)

## Reinforcement Learning & Algorithms

In Reinforcement Learning  (RL) an agent interacts with a given environment via state visiting, action selection. After every pair (S,a) of these, the environment returns a numerical reward (R) and then the agent transitions the next state (S'). Note that this transition may be entirely dependent on the 4 dimension vector (S,a,R,S') yielding a unique Markov Chain for each of the run episodes.

Developing and setting up the environment is a crucial yet not trivial part of any RL project. In this section, we will not provide this sort information; nonetheless, one can turn to section "PySC2 for Learning StarCraft2" for more details. Instead, here we will briefly review the Proximal Policy Optimization (PPO) algorithm which we chose to adapt the network parameters. Details on the network may be found in the following section. For the sake of clarity, we will not rely on the exact mathematical details but instead 
provide a heuristic vision of such algorithm. However, appropiate references will always be provided.

In RL (Sutton & Barto, 2018; Maxim Lapan, 2020), actions are selected with a given probability. Note that in each one of the environment's state (S) not all the actions may be available. For this reason, we shall define a policy as the probability of selecting an action (a) at given state (S). The goal of any RL algorithm is to addequately modify this policy in order to select the most optimal actions in each state of the environment, hence the name of Policy Optimization. 

In the PPO implementation (John Schulman et al, 2017), a reasonable quantity to backpropagate inside the network is the ratio between the actual policy and the previous one. This ratio, which is entirely dependent on the values for each one of the network parameters, provides an estimate of the change that the policy has undergone after a given update. However, this proxy does not take into account whether the proposed change in the policy was either good or bad. For this reason, we need to weight this ratio by the "Advantatge" of having changed the policy with respect to the provious one. 

As the name suggests, the Advantatge function contains the total reward obtained when the agent has followed the new policy. One must note that by just computing the rewards a perfectly valid proxy for evaluating the goodness of the change is easily available. But this approach does not take into account how good was the outcome based on previous experiences: a positive reward will always be good. Nonetheless, a positive reward may in fact be worse than previous rewards. To incorporate this, we also trained a neural network in charge of learning the value of the current policy. Since we now have a critic network, we can compare the goodness of the current outcome with the previous ones. In summary, the Advantatge function is now the difference between the received reward and the value of the critic. 

In addition, we used a more sophisticated version of this advantatge function called Generalized Advantatge Estimation (GAE). Nonetheless, the idea remains the same. This GAE has an optimal bias-variance trade-off which provides both nice stability and speed to the training progress. Detailed information may be found in John Schulman, 2017 and https://towardsdatascience.com/generalized-advantage-estimate-maths-and-code-b5d5bd3ce737.

To conclude, we also added an entropy term to be backpropagated in the network. In RL, it exists a fundamental problem called the "Exploration vs. Exploitation dilemma". Once the agent has a reasonably good policy, what is better, to keep exploting it or explore new states of the environment which may in the end report better outcomes? It has been shown (Haarnoja et al, 2017) that incorporating the entropy of the policy to the loss backpropagation helps to find an optimal solution to the afore mentioned dilemma. 

## Neural Network

![Screenshot 2021-07-12 at 19 57 14](https://user-images.githubusercontent.com/75299844/125630772-bc92ad42-810a-4f9e-bf5d-496b0205174d.png)

## Setup for development

1. Make sure you have Docker installed on your machine.
2. Install VSCode.
3. When you open the folder, it'll ask you to "Reopen in Container". Do it, and you're set!
4. You might need to re-run `pip install poetry` after starting the container.

### Training an agent

This will train a PPO agent for 1000 game steps (10M is more of a real training session) on MoveToBeacon and save replays, videos and the model in `./out`:

```
docker build aidl/betastar:latest .
docker run -it -v $PWD/out:/output aidl/betastar:latest train --agent full_ppo --environment SC2MoveToBeacon-v0 --output-path /output --total-steps 1000
```

When wandb asks you for an API key (that's why we need the `-it`), you can enter one or select the third option (Don't visualize my results).

### Playing with a trained agent

From the previous step, you'll have a model at `./out/RUN_ID/last_model.pth`. You can see it in action like this:

```
docker build aidl/betastar:latest .
docker run -it -v $PWD/out:/output aidl/betastar:latest play --model /output/RUN_ID/last_model.pth --episodes 4 --output-path /output
```

Replays and videos from this playtest will be under `./out` as well.

## Progress & Conclusions

Our main conclusion is that Reinforcement Learning is both fascinating and difficult. For the sake of claritty, we shall divide our progress thorught this project in three separate sections

#### 1. Preliminaires

During the first two months we focused on getting to know all the Reinforcement Learning background needed to face our main goal. We were able to understant several algorithms we later implemented to some benchmark problems of RL. This "toy" environments were basically CartPole, LunarLanding and MountainCar. We succesfully solved the first two. We were not able to validate our final algorithm implementation on MountainCar but we believe that this is due to the enormously number of training episodes requiered to solve the environment. Nonetheless, we have good reasons to believe that, given the appropiate amount of training time, we would have solved the it. In order to give a graphical description of the process, we present a gif containing the learning curves of all implemented algorithms in CartPole appearing in order of complexity. We must emphasize that, although Reinforce is very good at this particular environment, it is by far the least applicable in the majority of RL problems; including, of course, StarCraft 2. 

![CartPole](https://user-images.githubusercontent.com/75299844/125626475-879ae103-af62-409f-83a7-736c16ac5d08.gif)

#### 2. Setting the environment

Being confident we had deployable algorithms, we proceeded to setting up the StarCraft 2 environment. As mentioned earlier, we made use of PySC2 which already provides a wrapped set of methods specifically designed for our purpose. Far from being easy, we also had to invest a reasonable amount of time to dockerize everything so we could work on the project without any compatibility issue. One main drawback of PySC2 is the need to have StarCraft 2 installed. However, thanks to Docker, this was no longer required in our case. 

#### 3. Training the neural network

In the end, with all the previous work we were able to succesfully train a RL agent to play and win some minigames of StarCraft 2. Achieving this was not easy and in fact we should comment two significant breakthroughs that helped us. The first one was the idea of simplifying the available actions the agents could choose. Thanks to this, we were able to debug more easily some highly non-trivial mistakes both with our algotihm training process and with the environment setup. Once we saw that this simplified trained agent could win the "MoveToBeacon" minigame, we deleted these modifications and started once again with the complete agent and environment. The second breakthrough is the finding of a paper (Vinyals et al, 2017) in which the authors developed deep neural networks that could be trained to play the same minigames we were aming for. Nonetheless, the main goal of our project can be considered as achieved since we have been able to train a deep neural network with Reinforcement Learning methods that could eventually find and defeat Zerglings... Here is a video to prove it!

https://user-images.githubusercontent.com/75299844/125636947-028bf898-82ff-413b-bcb5-36a129e5f943.mp4

## Future Work

Despite considering ourselves highly satisfied with the results, it is true that we were not able to adress some key issues. For starters, we did not have time to do hyperparameter tunning. The trained agents do astonishingly well in some minigames, even reaching DeepMind's performance, but in some others it layed far from it. We must insist in that all trained agents showed coherent behavior but perhaps is was not optimal. We hypothesize that with some hyperparameter tunning, their scores would significantly improve.

And of course, we are forced to admit that in a real StarCraft 2 match, our agents would not stand a chance against any professional player human or binary...

## Acknowledgements

We are grateful to our advisor Dani Fojo for very helpful comments regarding the implementation of various algorithms as well as some really enlightening discussions involving Reinforcement Learning, physics, mathematics, philosophy and life. 

And finally, just remember that with the appropiate reward function you can train a monkey to behave like a human... and viceversa! 
