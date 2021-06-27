from typing import List, Tuple

import numpy as np
import torch as T
import wandb
from betastar.agents import base_agent
from betastar.data import Episode, UnrollDataset, collate
from betastar.envs.env import Action, ActionMask, PySC2Env, Value, spawn_env
from betastar.player import Player
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class Encoder(nn.Module):
    def __init__(
        self,
        screen_channels: int = 5,
        minimap_channels: int = 4,
        non_spatial_channels: int = 34,
        encoded_size: int = 128,
        spatial_dim: int = 16,
    ):
        conv_out_width = int(spatial_dim / 2 - 2)

        super().__init__()

        self.screen = nn.Sequential(
            Scale(1 / 255),
            nn.Conv2d(screen_channels, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * conv_out_width * conv_out_width, encoded_size),
            nn.ReLU(),
        )

        self.minimap = nn.Sequential(
            Scale(1 / 255),
            nn.Conv2d(minimap_channels, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * conv_out_width * conv_out_width, encoded_size),
            nn.ReLU(),
        )

        self.non_spatial = nn.Sequential(
            nn.Linear(non_spatial_channels, encoded_size), nn.ReLU()
        )

    def forward(self, screens, minimaps, non_spatials):
        a = self.screen(screens)
        b = self.minimap(minimaps)
        c = self.non_spatial(non_spatials)
        return T.cat([a, b, c]).view(screens.shape[0], -1)


class ActorCritic(nn.Module):
    def __init__(
        self, env: PySC2Env, encoded_size: int = 128, hidden_size: int = 128
    ) -> None:
        super().__init__()
        screen_channels: int = env.feature_original_shapes[0][0]  # type: ignore
        minimap_channels: int = env.feature_original_shapes[1][0]  # type: ignore
        non_spatial_channels: int = env.feature_original_shapes[2][0]  # type: ignore

        self.action_space = env.action_space.nvec.copy()  # type: ignore

        self.encoder = Encoder(
            encoded_size=encoded_size,
            screen_channels=screen_channels,
            minimap_channels=minimap_channels,
            non_spatial_channels=non_spatial_channels,
            spatial_dim=env.spatial_dim,
        )

        self.backbone = nn.Sequential(
            nn.Linear(encoded_size * 3, hidden_size), nn.ReLU()
        )

        self.actor = nn.Linear(hidden_size, self.action_space.sum())  # type: ignore

        self.critic = nn.Linear(hidden_size, 1)

    def forward(
        self, screens, minimaps, non_spatials, action_mask: ActionMask
    ) -> Tuple[Action, Value]:
        latent = self.encode(screens, minimaps, non_spatials)
        return self.act(latent, action_mask=action_mask), self.critic(latent.detach())

    def encode(self, screens, minimaps, non_spatials) -> Tensor:
        return self.backbone(self.encoder(screens, minimaps, non_spatials))

    def act(self, latent, action_mask: ActionMask) -> Action:
        logits = self._masked_logits(self.actor(latent), action_mask)
        categoricals = self.discrete_categorical_distributions(logits)
        return T.stack([categorical.sample() for categorical in categoricals], dim=1)

    def _masked_logits(self, logits: Tensor, action_mask: ActionMask) -> Tensor:
        """
        Filters the logits according to the environment's action mask.
        This ensures that actions that are not currently available will
        never be sampled.
        """
        return T.where(action_mask.bool(), logits, T.tensor(-1e8))

    def discrete_categorical_distributions(self, logits: Tensor) -> List[Categorical]:
        """
        Split the flat logits into the subsets that represent the many
        categorical distributions (actions, screen coordinates, minimap coordinates, etc...).
        Returns a Categorical object for each of those.
        """
        split_logits = T.split(logits, self.action_space.tolist(), dim=1)  # type: ignore
        return [Categorical(logits=logits) for logits in split_logits]


def _compute_unroll_returns(rewards: Tensor, bootstrap: float, config: wandb.Config):
    """
    Computes returns on a single unroll of arbitrary length.
    """
    returns = []
    R = bootstrap
    for r in reversed(rewards):
        R = r + R * config.reward_decay
        returns.insert(0, R)

    returns = T.tensor(returns)
    if config.normalize_returns:
        return (returns - returns.mean()) / (
            returns.std() + np.finfo(np.float32).eps.item()
        )
    else:
        return returns


def compute_returns(
    rewards: Tensor, next_values: Tensor, dones: Tensor, config: wandb.Config
):
    """
    Computes returns on a series of unrolls.
    """
    unroll_rewards = rewards.split(config.unroll_length)
    unroll_next_values = next_values.split(config.unroll_length)
    unroll_dones = dones.split(config.unroll_length)

    batch_returns = []
    for (u_rewards, u_next_values, u_dones) in zip(
        unroll_rewards, unroll_next_values, unroll_dones
    ):
        if u_dones[-1]:
            bootstrap = 0.0
        else:
            bootstrap = u_next_values[-1].item()
        batch_returns.append(_compute_unroll_returns(u_rewards, bootstrap, config))

    return T.cat(batch_returns)


def compute_advantages(
    rewards: Tensor, values: Tensor, next_value: float, config: wandb.Config
):
    advantages = []
    advantage = 0

    for r, v in zip(reversed(rewards), reversed(values)):
        td_error = r + next_value * config.reward_decay - v
        advantage = td_error + advantage * config.reward_decay * config.gae_lambda
        next_value = v
        advantages.insert(0, advantage)

    advantages = T.stack(advantages)
    if config.normalize_advantages:
        return (advantages - advantages.mean()) / (
            advantages.std() + np.finfo(np.float32).eps.item()
        )
    else:
        return advantages


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    dones: Tensor,
    config: wandb.Config,
) -> Tuple[Tensor, Tensor]:
    N = len(rewards)
    advantages = T.zeros_like(rewards).float()
    lastgaelam = 0.0
    for t in reversed(range(N)):
        nextnonterminal = 1.0 - dones[t].item() # type: ignore
        nextvalues = next_values[t]

        delta = (
            rewards[t] + config.reward_decay * nextvalues * nextnonterminal - values[t]
        )
        advantages[t] = lastgaelam = (
            delta
            + config.reward_decay * config.gae_lambda * nextnonterminal * lastgaelam
        )
    returns = advantages + values
    return advantages, returns


class A2CPlayer(Player):
    def __init__(self, model: ActorCritic) -> None:
        super(A2CPlayer).__init__()
        self.model = model

    def act(
        self, screen: Tensor, minimap: Tensor, non_spatial: Tensor, action_mask: Tensor
    ) -> Tensor:
        latents = self.model.encode(screen, minimap, non_spatial)
        return self.model.act(
            latents,
            action_mask=action_mask,
        )

    def evaluate(self, screen: Tensor, minimap: Tensor, non_spatial: Tensor) -> Tensor:
        latents = self.model.encode(screen, minimap, non_spatial)
        return self.model.critic(latents)

    def reload_from(self, model: nn.Module):
        self.model.load_state_dict(model.state_dict())  # type: ignore
        self.model.eval()


class A2C(base_agent.BaseAgent):
    def dataloader(self, episodes: List[Episode]) -> DataLoader:
        return DataLoader(
            UnrollDataset(episodes, self.config.unroll_length),
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate,
        )

    def run(self):
        device = T.device("cuda" if T.cuda.is_available() else "cpu")  # type: ignore

        env = spawn_env(self.config.environment, self.config.game_speed, rank=-1)
        model = ActorCritic(env=env).to(device)
        model.train()

        max_reward = 0

        if self.config.use_ppo:
            previous_model = ActorCritic(env=env).to(device)
            previous_model.load_state_dict(model.state_dict())  # type: ignore

        player = A2CPlayer(ActorCritic(env=env))

        env.close()

        opt = Adam(
            model.parameters(), lr=self.config.learning_rate, betas=(0.92, 0.999)
        )

        wandb.watch(model, log="all")

        cycles = 0

        step_n = 0
        with tqdm(range(self.config.total_steps), unit="steps", leave=False) as pbar:
            while step_n < self.config.total_steps:
                player.reload_from(model)

                episodes, episode_rewards = self.play(player)

                real_losses = []
                actor_losses = []
                critic_losses = []
                entropies = []

                for _update_epoch in tqdm(
                    range(self.config.update_epochs), unit="epochs", leave=False
                ):
                    with tqdm(
                        self.dataloader(episodes), unit="batches", leave=False
                    ) as batches:
                        for (
                            screens,
                            minimaps,
                            non_spatials,
                            actions,
                            rewards,
                            _values,
                            next_values,
                            action_masks,
                            dones,
                        ) in batches:
                            screens = screens.to(device)
                            minimaps = minimaps.to(device)
                            non_spatials = non_spatials.to(device)
                            actions = actions.to(device)
                            rewards = rewards.to(device)
                            next_values = next_values.to(device)
                            action_masks = action_masks.to(device)
                            dones = dones.to(device)

                            latents = model.encode(screens, minimaps, non_spatials)
                            logits = T.where(
                                action_masks.bool(), model.actor(latents), T.tensor(-1e8).to(device)
                            )

                            values = model.critic(latents.detach()).squeeze(dim=1)

                            categoricals = model.discrete_categorical_distributions(logits)
                            log_probs = T.stack(
                                [
                                    categorical.log_prob(a)
                                    for a, categorical in zip(
                                        actions.transpose(0, 1), categoricals
                                    )
                                ],
                                dim=1,
                            )

                            if self.config.use_ppo:
                                old_categoricals = previous_model.discrete_categorical_distributions(logits)  # type: ignore
                                old_log_probs = T.stack(
                                    [
                                        categorical.log_prob(a)
                                        for a, categorical in zip(
                                            actions.transpose(0, 1), old_categoricals
                                        )
                                    ],
                                    dim=1,
                                )

                                # PPO Clip
                                # do we want to aggregate first and then calculate a big ratio?
                                ratio = T.exp(log_probs.sum(dim=1) - old_log_probs.sum(dim=1))

                                clipped_ratio = ratio.clamp(
                                    min=1.0 - self.config.clip_range,
                                    max=1.0 + self.config.clip_range,
                                )

                                # or do we want to calculate ratios to clip independently, and then aggregate?
                                # ratio = T.exp(log_probs - old_log_probs)

                                # clipped_ratio = ratio.clamp(min=1.0 - self.config.clip_range,
                                #                             max=1.0 + self.config.clip_range, dim=1)

                                # ratio = ratio.sum(dim=1)
                                # clipped_ratio = clipped_ratio.sum(dim=1

                            entropy = T.stack(
                                [categorical.entropy().mean() for categorical in categoricals]
                            ).mean()

                            if self.config.use_gae:
                                advantages, returns = compute_gae(
                                    rewards, values, next_values, dones, self.config
                                )
                            else:
                                returns = compute_returns(
                                    rewards,
                                    next_values,
                                    dones,
                                    self.config,
                                )
                                advantages = returns - values

                            if self.config.use_ppo:
                                actor_loss = T.min(
                                    ratio * advantages.detach(),  # type: ignore
                                    clipped_ratio * advantages.detach(),  # type: ignore
                                )  # type: ignore
                                actor_loss = (-actor_loss).mean()  # type: ignore
                            else:
                                actor_loss = (
                                    -log_probs.sum(dim=1) * advantages.detach()
                                ).mean()
                            critic_loss = advantages.pow(2).mean() * self.config.critic_coeff
                            entropy_loss = entropy * self.config.entropy_coeff
                            loss = actor_loss + critic_loss - entropy_loss

                            batches.set_postfix(
                                {
                                    "loss": loss.item(),
                                    "al": actor_loss.item(),
                                    "cl": critic_loss.item(),
                                    "ent": entropy_loss.item(),
                                }
                            )

                            opt.zero_grad()
                            loss.backward()

                            if self.config.use_ppo:
                                # save model as previous model
                                previous_model.load_state_dict(model.state_dict())  # type: ignore

                            opt.step()

                            real_losses.append(loss.item())
                            actor_losses.append(actor_loss.item())
                            critic_losses.append(critic_loss.item())
                            entropies.append(entropy_loss.item())

                _max_reward = np.array(episode_rewards).max()
                metrics = {
                    "episode_reward/mean": np.array(episode_rewards).mean(),
                    "episode_reward/max": _max_reward,
                    "loss/real": np.array(real_losses).mean(),
                    "loss/actor": np.array(actor_losses).mean(),
                    "loss/critic": np.array(critic_losses).mean(),
                    "loss/entropy": np.array(entropies).mean()
                }
                if cycles % 80 == 0 or step_n > self.config.total_steps:
                    metrics['video'] = self.last_video(step_n)

                if _max_reward > max_reward:
                    wandb.run.summary["episode_reward/max"] = _max_reward
                    max_reward = _max_reward
                wandb.log(
                    metrics,
                    step=step_n,
                    commit=True
                )
                if cycles % 80 == 0 or step_n > self.config.total_steps:
                    wandb.log_artifact(
                        self.last_replay(map_name=self.config.environment, step_n=step_n)
                    )
                cycles += 1

                _played_steps = sum([len(e) for e in episodes])
                pbar.update(_played_steps)
                step_n += _played_steps

        wandb.finish()
        self.env.stop()
