import torch
from torch import nn
from torch.nn import functional as tf


class JointLoss(nn.Module):
    def __init__(self, balance: float = 1):
        super().__init__()
        self._balance = balance

    def forward(
        self,
        ent_logit: torch.Tensor,
        obj: torch.Tensor,
        rel_logit: torch.Tensor,
        rel: torch.Tensor,
    ):
        rel_loss = tf.cross_entropy(rel_logit, rel, reduction="none")
        ent_loss = tf.cross_entropy(ent_logit, obj, reduction="none")
        return ent_loss + self._balance * rel_loss
