import copy
import enum
from typing import DefaultDict, Optional

import torch
import torchmetrics
from torch import nn
from torch.nn import functional as tf
from torchmetrics import functional as tmf
from torchmetrics.collections import MetricCollection

from . import datasets, models

__all__ = ["datasets", "models"]


class RelationPredictionCriterion(nn.Module):
    def forward(self, rel_logit: torch.Tensor, rel: torch.Tensor):
        return tf.cross_entropy(rel_logit, rel, reduction="none")


class NodePredictionCriterion(nn.Module):
    def forward(self, tail_logit: torch.Tensor, tail: torch.Tensor):
        return tf.cross_entropy(tail_logit, tail, reduction="none")


class TKGCriterion(nn.Module):
    def forward(
        self,
        tail_logit: torch.Tensor,
        tail: torch.Tensor,
        rel_logit: torch.Tensor,
        rel: torch.Tensor,
    ):
        rel_loss = tf.cross_entropy(rel_logit, rel, reduction="none")
        ent_loss = tf.cross_entropy(tail_logit, tail, reduction="none")
        return rel_loss + ent_loss


class MRR(torchmetrics.MeanMetric):
    def update(self, logit: torch.Tensor, target: torch.Tensor) -> None:
        value = tmf.retrieval_reciprocal_rank(logit, target)
        return super().update(value)


class Hit(torchmetrics.MeanMetric):
    def __init__(self, k: int):
        super().__init__()
        self._k = k

    def update(self, logit: torch.Tensor, target: torch.Tensor) -> None:
        value = tmf.retrieval_hit_rate(logit, target, k=self._k)
        return super().update(value)


class FilterMode(enum.Enum):
    raw = enum.auto()
    time = enum.auto()


class TKGMetric(torchmetrics.Metric):
    def __init__(self) -> None:
        metrics = {
            "mrr": MRR(),
            "hit@1": Hit(k=1),
            "hit@3": Hit(k=3),
            "hit@10": Hit(k=10),
        }
        super().__init__()
        self._raw = MetricCollection(
            metrics=copy.deepcopy(metrics), prefix="r_"
        )
        self._time = MetricCollection(
            metrics=copy.deepcopy(metrics), prefix="t_"
        )

    def _inner_update(
        self,
        logit: torch.Tensor,
        time_logit: torch.Tensor,
        target: torch.Tensor,
    ):
        raw = self._raw(logit, target)
        time = self._time(time_logit, target)
        return {**raw, **time}

    def compute(self):
        ret = {}
        ret.update(self._raw.compute())
        ret.update(self._time.compute())
        return ret


class RelPredictionMetric(TKGMetric):
    def update(
        self,
        rel_logit: torch.Tensor,
        rel: torch.Tensor,
        head: torch.Tensor,
        tail: torch.Tensor,
    ) -> None:
        """
        Arguments:
            rel_logit: (batch_size, num_relations)
            rel: (batch_size,)
        """
        target = tf.one_hot(rel, rel_logit.size(-1))

        r_dict = DefaultDict(list)
        for r, h, t in zip(rel, head, tail):
            r_dict[(h.item(), t.item())].append(r.item())

        mask = torch.zeros_like(rel_logit, dtype=torch.bool)
        for rs in r_dict.values():
            mask[rs, rs] = 1

        time_logit = rel_logit.masked_fill(mask, 0)
        time_logit = torch.where(target != 0, rel_logit, time_logit)

        return self._inner_update(rel_logit, time_logit, target)


class TailPredictionMetric(TKGMetric):
    def update(self, tail_logit: torch.Tensor, tail: torch.Tensor) -> None:
        if self.filter == FilterMode.raw:
            return super().update(tail, tf.one_hot(tail_logit.size(-1)))
        elif self.filter == FilterMode.time:
            raise
        else:
            raise AttributeError()


def tkg_metric_update(
    metric: TKGMetric,
    logit: torch.Tensor,
    target: torch.Tensor,
):
    if metric.filter == FilterMode.raw:
        return metric.update(target, tf.one_hot(logit.size(-1)))
    elif metric.filter == FilterMode.time:
        raise
    else:
        raise AttributeError()
    pass
