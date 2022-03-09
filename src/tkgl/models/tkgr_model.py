import abc
import logging
from typing import Dict, List, Tuple, Type

import dgl
import molurus
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


class JointSigmoidLoss(torch.nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self._alpha = alpha

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
        return ent_loss * self._alpha + rel_loss * (1 - self._alpha)


class TkgrModel(torch.nn.Module):

    num_ents: int
    num_rels: int
    hidden_size: int

    rel_score: RelScoreFunction
    obj_score: NodeScoreFunction

    @abc.abstractmethod
    def evolve(
        self, hist_graphs: List[dgl.DGLGraph]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @classmethod
    def build_criterion(cls, alpha: float):
        return JointLoss(alpha)


def build_model(cfg: Dict, **kwargs) -> TkgrModel:
    model_arch = cfg.pop("arch")
    model_class: Type[TkgrModel] = molurus.import_get(model_arch)

    if "TemporalRerank" in model_arch:
        backbone_cfg = cfg.pop("backbone")
        for k, v in cfg.items():
            dict.setdefault(backbone_cfg, k, v)

        cfg["backbone"] = build_model(backbone_cfg, **kwargs)

    logger.info(f"Model: {model_class.__name__}")
    model = molurus.smart_call(model_class, cfg, **kwargs)
    return model
