from betastar.envs.env import Value
from typing import List, Tuple
import torch

class ScreenNet(torch.nn.Module):
    """Model for movement based mini games in sc2.
    This network only takes screen input and only returns spatial outputs.
    Some of the example min games are MoveToBeacon and CollectMineralShards.
    Arguments:
        - in_channel: Number of feature layers in the screen input
        - screen_size: Screen size of the mini game. If 64 is given output
            size will be 64*64
    Note that output size depends on screen_size.
    """

    class ResidualConv(torch.nn.Module):
        def __init__(self, in_channel, **kwargs):
            super().__init__()
            assert kwargs["out_channels"] == in_channel, (
                "input channel must" "be the same as out" " channels"
            )
            self.block = torch.nn.Sequential(
                torch.nn.Conv2d(**kwargs),
                torch.nn.InstanceNorm2d(in_channel),
                torch.nn.ReLU(),
                torch.nn.Conv2d(**kwargs),
                torch.nn.InstanceNorm2d(in_channel),
                torch.nn.ReLU(),
            )

        def forward(self, x):
            res_x = self.block(x)
            return res_x + x

    def __init__(self, in_channel, screen_size):
        super().__init__()
        res_kwargs = {
            "in_channels": 32,
            "out_channels": 32,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
        }
        self.convnet = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, 32, 3, 1, padding=1),
            self.ResidualConv(32, **res_kwargs),
            self.ResidualConv(32, **res_kwargs),
            self.ResidualConv(32, **res_kwargs),
            self.ResidualConv(32, **res_kwargs),
        )

        self.policy = torch.nn.Sequential(
            torch.nn.Linear(32 * screen_size * screen_size, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2 * screen_size),
        )

        self.value = torch.nn.Sequential(
            torch.nn.Linear(32 * screen_size * screen_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

        self.screen_size = screen_size
        gain = torch.nn.init.calculate_gain("relu")

        def param_init(module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight, gain)
                torch.nn.init.zeros_(module.bias)
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.dirac_(module.weight)
                torch.nn.init.zeros_(module.bias)  # type: ignore

        self.apply(param_init)


    def forward(
        self, screens, _minimaps, _non_spatials, _action_mask
    ) -> Tuple[List[torch.Tensor], Value]:
        encode = self.convnet(screens / 255)
        encode = encode.reshape(encode.shape[0], -1)

        value = self.value(encode)

        logits = self.policy(encode)

        return logits.split(self.screen_size, dim=-1), value