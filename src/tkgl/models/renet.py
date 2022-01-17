import torch
from torch import nn


class RecurrentGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self._encoder = None

    def forward(self):
        pass


class RGCN(nn.Module):
    pass


class RGCNPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ent_embed: torch.Tensor, rel_embed: torch.Tensor):
        pass


class AttentivePooling(nn.Module):
    pass
