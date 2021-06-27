from dataclasses import dataclass, field
from typing import List, Tuple

import torch as T
from torch import Tensor
from torch.utils.data import Dataset

from betastar.envs.env import Action, ActionMask, Observation, Reward, Value


@dataclass
class Episode:
    states: List[Observation] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    rewards: List[Reward] = field(default_factory=list)
    values: List[Value] = field(default_factory=list)
    next_values: List[Value] = field(default_factory=list)
    action_masks: List[ActionMask] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    count: int = 0

    def __repr__(self) -> str:
        return f"#({self.count} steps, {sum(self.rewards)} reward)"

    def __len__(self):
        return self.count

    def lookup(self, idxs):
        return [self[idx] for idx in idxs]

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.values[idx],
            self.next_values[idx],
            self.action_masks[idx],
            self.dones[idx],
        )

    def add(
        self,
        observation: Observation,
        action: Action,
        reward: Reward,
        value: Value,
        next_value: Value,
        action_mask: ActionMask,
        done: bool,
    ):
        self.states.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.next_values.append(next_value)
        self.action_masks.append(action_mask)
        self.dones.append(done)
        self.count += 1


class UnrollDataset(Dataset):
    """
    Flatten episodes into neat (at-most) unroll-size chunks.
    """

    def __init__(self, episodes: List[Episode], unroll_size: int) -> None:
        super().__init__()
        self.unrolls = []
        for e in episodes:
            idx_chunks = T.arange(0, len(e)).long().split(unroll_size)
            for chunk in idx_chunks:
                if len(chunk) == unroll_size:
                    self.unrolls.append(e.lookup(chunk))
                else:
                    # get last len(chunk) elements of episode
                    self.unrolls.append(e.lookup(list(range(-(len(chunk)), 0))))

    def __len__(self):
        return len(self.unrolls)

    def __getitem__(self, idx):
        return self.unrolls[idx]


def collate(batch):
    screens = T.stack([step[0][0] for trajectory in batch for step in trajectory])
    minimaps = T.stack([step[0][1] for trajectory in batch for step in trajectory])
    non_spatials = T.stack([step[0][2] for trajectory in batch for step in trajectory])
    actions = T.stack([step[1] for trajectory in batch for step in trajectory], dim=0)
    rewards = T.tensor([step[2] for trajectory in batch for step in trajectory])
    values = T.tensor([step[3] for trajectory in batch for step in trajectory])
    next_values = T.tensor([step[4] for trajectory in batch for step in trajectory])
    action_masks = T.stack([step[5] for trajectory in batch for step in trajectory])
    dones = T.tensor([step[6] for trajectory in batch for step in trajectory])

    return (
        screens,
        minimaps,
        non_spatials,
        actions,
        rewards,
        values,
        next_values,
        action_masks,
        dones,
    )
