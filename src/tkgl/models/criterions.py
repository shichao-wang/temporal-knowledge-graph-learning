import torch
from torch import nn
from torch.nn import functional as tf


class JointLoss(nn.Module):
    def __init__(self, balance: float = 0.5):
        super().__init__()
        self._balance = balance

    def forward(
        self,
        obj_logit: torch.Tensor,
        obj: torch.Tensor,
        rel_logit: torch.Tensor,
        rel: torch.Tensor,
    ):
        rel_loss = tf.cross_entropy(rel_logit, rel, reduction="none")
        ent_loss = tf.cross_entropy(obj_logit, obj, reduction="none")
        return ent_loss * self._balance + rel_loss * (1 - self._balance)


class EntLoss(nn.Module):
    def forward(self, obj_logit: torch.Tensor, obj: torch.Tensor):
        return tf.cross_entropy(obj_logit, obj, reduction="none")


class RefineLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self._loss_ori = JointLoss(alpha)
        self._loss_re = JointLoss(alpha)
        self._beta = beta

    def forward(
        self,
        obj_logit: torch.Tensor,
        obj_logit_ori: torch.Tensor,
        obj: torch.Tensor,
        rel_logit: torch.Tensor,
        rel_logit_ori: torch.Tensor,
        rel: torch.Tensor,
    ):
        losses_ori = self._loss_ori(obj_logit_ori, obj, rel_logit_ori, rel)
        losses_re = self._loss_re(obj_logit, obj, rel_logit, rel)

        return losses_re * self._beta + losses_ori * (1 - self._beta)
