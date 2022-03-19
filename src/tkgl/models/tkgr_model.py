import abc
import logging
from typing import Callable, List, Tuple, TypedDict

import dgl
import torch
from torch.nn import functional as tf

from tkgl.scores import NodeScoreFunction, RelScoreFunction

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


class TkgrReturns(TypedDict):
    ent_embeds: torch.Tensor
    rel_embeds: torch.Tensor
    obj_logit: torch.Tensor
    rel_logit: torch.Tensor


class TkgrModel(torch.nn.Module):

    num_ents: int
    num_rels: int
    hidden_size: int

    rel_score: RelScoreFunction
    obj_score: NodeScoreFunction

    __call__: Callable[..., TkgrReturns]

    @abc.abstractmethod
    def evolve(
        self, hist_graphs: List[dgl.DGLGraph]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @classmethod
    def build_criterion(cls, alpha: float):
        return JointLoss(alpha)
