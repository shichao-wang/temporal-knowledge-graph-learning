import torch


class TemporalRerank(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        self._backbone = backbone
