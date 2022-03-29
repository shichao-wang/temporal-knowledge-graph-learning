import logging

import torch
from torch.nn import functional as tf

logger = logging.getLogger(__name__)


class JointLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self._alpha = alpha

    def forward(
        self,
        obj_logit: torch.Tensor,
        obj: torch.Tensor,
        rel_logit: torch.Tensor,
        rel: torch.Tensor,
    ):
        rel_loss = tf.cross_entropy(rel_logit, rel)
        ent_loss = tf.cross_entropy(obj_logit, obj)
        return ent_loss * self._alpha + rel_loss * (1 - self._alpha)


class EntLoss(torch.nn.Module):
    def forward(self, obj_logit: torch.Tensor, obj: torch.Tensor):
        ent_loss = tf.cross_entropy(obj_logit, obj)
        return ent_loss


class TkgrModel(torch.nn.Module):
    def __init__(
        self,
        num_ents: int,
        num_rels: int,
        hidden_size: int,
    ):
        super().__init__()
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.hidden_size = hidden_size
        self.ent_emb = torch.nn.Parameter(torch.empty(num_ents, hidden_size))
        self.rel_emb = torch.nn.Parameter(torch.empty(num_rels, hidden_size))

        torch.nn.init.normal_(self.ent_emb)
        torch.nn.init.xavier_normal_(self.rel_emb)
        # torch.nn.init.normal_(self.ent_emb)
        # torch.nn.init.xavier_normal_(self.rel_emb)
