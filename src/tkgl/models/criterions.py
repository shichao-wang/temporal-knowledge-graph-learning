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
        rel_loss = tf.cross_entropy(rel_logit, rel)
        ent_loss = tf.cross_entropy(obj_logit, obj)
        return ent_loss * self._balance + rel_loss * (1 - self._balance)


class JointSigmoidLoss(torch.nn.Module):
    def __init__(self, balance: float):
        super().__init__()
        self._balance = balance

    def forward(
        self,
        obj_logit: torch.Tensor,
        obj: torch.Tensor,
        rel_logit: torch.Tensor,
        rel: torch.Tensor,
    ):
        rel_loss = tf.binary_cross_entropy_with_logits(
            rel_logit, tf.one_hot(rel, rel_logit.size(-1))
        )
        ent_loss = tf.binary_cross_entropy_with_logits(
            obj_logit, tf.one_hot(obj, obj_logit.size(-1))
        )
        # rel_loss = tf.cross_entropy(rel_logit, rel)
        # ent_loss = tf.cross_entropy(obj_logit, obj)
        return ent_loss * self._balance + rel_loss * (1 - self._balance)


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
