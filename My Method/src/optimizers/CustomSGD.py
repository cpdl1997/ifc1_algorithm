from typing import Any, Dict
import torch
from torch import Tensor
from torch.optim.sgd import SGD
from collections.abc import Iterable

class CustomSGD(SGD):

    def __init__(self, params: Iterable[Tensor] | Iterable[Dict[str, Any]], lr: float = ..., momentum: float = ..., dampening: float = ..., weight_decay: float = ..., nesterov: bool = ...) -> None:
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def step():
        return