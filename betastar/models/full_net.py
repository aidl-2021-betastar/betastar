from betastar.envs.env import Action, ActionMask, Value
from typing import List, Tuple
import torch as T
import torch.nn as nn
from torch.distributions import Categorical
from torch import Tensor

class ResidualConv(nn.Module):
    def __init__(self, in_channel, **kwargs):
        super().__init__()
        assert kwargs["out_channels"] == in_channel, (
            "input channel must" "be the same as out" " channels"
        )
        self.block = nn.Sequential(
            nn.Conv2d(**kwargs),
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(**kwargs),
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        res_x = self.block(x)
        return res_x + x

class Encoder(nn.Module):
    def __init__(
        self,
        screen_channels: int = 5,
        minimap_channels: int = 4,
        non_spatial_channels: int = 34,
        non_spatial_size: int = 32
    ):
        super().__init__()

        res_kwargs = {
            "in_channels": 32,
            "out_channels": 32,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
        }
        self.screen = nn.Sequential(
            nn.Conv2d(screen_channels, 32, 3, 1, padding=1),
            ResidualConv(32, **res_kwargs),
            ResidualConv(32, **res_kwargs),
            ResidualConv(32, **res_kwargs),
            ResidualConv(32, **res_kwargs),
        )

        self.minimap = nn.Sequential(
            nn.Conv2d(minimap_channels, 32, 3, 1, padding=1),
            ResidualConv(32, **res_kwargs),
            ResidualConv(32, **res_kwargs),
            ResidualConv(32, **res_kwargs),
            ResidualConv(32, **res_kwargs),
        )

        self.non_spatial = nn.Sequential(
            nn.Linear(non_spatial_channels, non_spatial_size), nn.ReLU()
        )

    def forward(self, screens, minimaps, non_spatials):
        a = self.screen(screens / 255)
        b = self.minimap(minimaps / 255)
        c = self.non_spatial(non_spatials.tanh()) # we take the tanh of those bc they get really big
        return T.cat([a, b], dim=1), c


class FullNet(nn.Module):
    def __init__(
        self, env, hidden_size: int = 256, non_spatial_size: int = 32
    ) -> None:
        super().__init__()
        screen_channels: int = env.feature_original_shapes[0][0]  # type: ignore
        minimap_channels: int = env.feature_original_shapes[1][0]  # type: ignore
        non_spatial_channels: int = env.feature_original_shapes[2][0]  # type: ignore

        self.action_space = env.action_space.nvec.copy()  # type: ignore

        self.encoder = Encoder(
            non_spatial_size=non_spatial_size,
            screen_channels=screen_channels,
            minimap_channels=minimap_channels,
            non_spatial_channels=non_spatial_channels
        )

        self.backbone = nn.Sequential(
            nn.Linear(64 * env.spatial_dim * env.spatial_dim + non_spatial_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        self.spatial_args_mask = env.spatial_args_mask
        spatial_args = len(env.spatial_args_mask[env.spatial_args_mask == 1])
        self.spatial_policy = nn.Linear(hidden_size, spatial_args)
        self.non_spatial_policy = nn.Linear(hidden_size, self.action_space.sum() - spatial_args)

        self.critic = nn.Linear(hidden_size, 1)

    def forward(
        self, screens, minimaps, non_spatials, action_mask: ActionMask
    ) -> Tuple[List[Tensor], Value]:
        latent_spatial, latent_non_spatial = self.encoder(screens, minimaps, non_spatials)
        latent = self.backbone(T.cat([latent_spatial.view(latent_spatial.shape[0], -1), latent_non_spatial], dim=1))
        logits = T.zeros((screens.shape[0], self.action_space.sum())).to(screens.device)

        spatial_mask = self.spatial_args_mask.to(screens.device)
        spatial_policy = self.spatial_policy(latent)
        non_spatial_policy = self.non_spatial_policy(latent)
        for i in range(screens.shape[0]):
            logits[i, spatial_mask.bool()] = spatial_policy[i]
            logits[i, ~spatial_mask.bool()] = non_spatial_policy[i]

        logits = self.split_logits(self._masked_logits(logits, action_mask))
        value = self.critic(latent.detach())
        return logits, value

    # def encode(self, screens, minimaps, non_spatials) -> Tensor:
    #     return self.backbone(self.encoder(screens, minimaps, non_spatials))

    # def act(self, latent, action_mask: ActionMask) -> Action:
    #     logits = self._masked_logits(self.actor(latent), action_mask)
    #     categoricals = self.discrete_categorical_distributions(logits)
    #     return T.stack([categorical.sample() for categorical in categoricals], dim=1)

    def _masked_logits(self, logits: Tensor, action_mask: ActionMask) -> Tensor:
        """
        Filters the logits according to the environment's action mask.
        This ensures that actions that are not currently available will
        never be sampled.
        """
        return T.where(action_mask.bool(), logits, T.tensor(-1e8).to(logits.device))

    def split_logits(self, logits: Tensor) -> List[Tensor]:
        return T.split(logits, self.action_space.tolist(), dim=1)  # type: ignore

    # def discrete_categorical_distributions(self, logits: Tensor) -> List[Categorical]:
    #     """
    #     Split the flat logits into the subsets that represent the many
    #     categorical distributions (actions, screen coordinates, minimap coordinates, etc...).
    #     Returns a Categorical object for each of those.
    #     """
    #     return [Categorical(logits=logits) for logits in self.split_logits(logits)]