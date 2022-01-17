import copy
import enum
from typing import DefaultDict, Dict, Hashable, List, Optional

import torch
import torchmetrics
from torch.nn import functional as tf
from torchmetrics import functional as tmf


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


class TKGBaseMetric(torchmetrics.MetricCollection):
    def __init__(self, mrr: bool = True, hits: bool = False) -> None:
        metrics = {}
        if mrr:
            metrics["mrr"] = MRR()
        if hits:
            for k in [1, 3, 10]:
                metrics["hit@{k}"] = Hit(k)
        super().__init__(metrics)


class EntMetric(TKGBaseMetric):
    def update(self, ent_logit: torch.Tensor, subj: torch.Tensor):
        onehot_target = tf.one_hot(subj, ent_logit.size(-1))
        return super().update(logit=ent_logit, target=onehot_target)


class RelMetric(TKGBaseMetric):
    def update(self, rel_logit: torch.Tensor, rel: torch.Tensor):
        onehot_target = tf.one_hot(rel, rel_logit.size(-1))
        return super().update(logit=rel_logit, target=onehot_target)


class JointMetric(torchmetrics.Metric):
    def __init__(self, mrr: bool = True, hits: bool = False) -> None:
        super().__init__()
        self._rel = RelMetric(mrr, hits)
        self._ent = EntMetric(mrr, hits)

    def update(
        self,
        ent_logit: torch.Tensor,
        rel_logit: torch.Tensor,
        subj: torch.Tensor,
        rel: torch.Tensor,
    ):
        self._rel.update(rel_logit, rel)
        self._ent.update(ent_logit, subj)

    def compute(self):
        ret = {}
        ret.update({"r_" + k: v for k, v in self._rel.compute().items()})
        ret.update({"e_" + k: v for k, v in self._ent.compute().items()})
        return ret


def time_aware_mask(key1, key2, value):
    d_dict = DefaultDict(list)
    for k1, k2, v in zip(key1, key2, value):
        d_dict[(k1.item(), k2.item())].append(v.item())
    return d_dict


def time_aware_filter(
    logit: torch.Tensor,
    target: torch.Tensor,
    d_dict: Dict[Hashable, List[torch.Tensor]],
):
    target = tf.one_hot(target, logit.size(-1))

    mask = torch.zeros_like(logit, dtype=torch.bool)
    for rs in d_dict.values():
        mask[rs, rs] = 1

    t_logit = logit.masked_fill(mask, 0)
    t_logit = torch.where(target != 0, logit, t_logit)
    return t_logit
