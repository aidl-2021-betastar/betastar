from abc import ABC, abstractmethod
from torch import Tensor

class Player(ABC):
    @abstractmethod 
    def act(self, screen: Tensor, minimap: Tensor, non_spatial: Tensor, action_mask: Tensor) -> Tensor: ...

    @abstractmethod 
    def evaluate(self, screen: Tensor, minimap: Tensor, non_spatial: Tensor) -> Tensor: ...
