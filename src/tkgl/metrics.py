from typing import DefaultDict, Dict, Hashable, List

import torch
import torchmetrics
from torch.nn import functional as tf


class RankMetric(torchmetrics.Metric):
    ranks: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state("ranks", torch.tensor([]))

    def update(self, logit: torch.Tensor, target: torch.Tensor):
        ranks = logit_ranks(logit, target)
        self.ranks = torch.cat([self.ranks, ranks], dim=-1)


def logit_ranks(logit: torch.Tensor, target: torch.Tensor):
    total_ranks = logit.argsort(dim=-1, descending=True)
    ranks = torch.nonzero(total_ranks == target.view(-1, 1))[:, 1]
    return ranks


def mrr_value(ranks: torch.Tensor):
    return torch.mean(1.0 / (ranks.float() + 1))


def hit_value(ranks: torch.Tensor, k: int):
    return (ranks < k).float().mean()


class MRR(RankMetric):
    def compute(self):
        return mrr_value(self.ranks)


class Hit(RankMetric):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def compute(self):
        return hit_value(self.ranks, self.k)


class EntMRR(MRR):
    def update(self, obj_logit: torch.Tensor, obj: torch.Tensor) -> None:
        return super().update(logit=obj_logit, target=obj)


class JointMetric(torchmetrics.Metric):
    ent_ranks: torch.Tensor
    rel_ranks: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state("ent_ranks", torch.tensor([]))
        self.add_state("rel_ranks", torch.tensor([]))

    def update(
        self,
        obj_logit: torch.Tensor,
        rel_logit: torch.Tensor,
        obj: torch.Tensor,
        rel: torch.Tensor,
    ):
        ent_ranks = logit_ranks(obj_logit, obj)
        self.ent_ranks = torch.cat([self.ent_ranks, ent_ranks], dim=-1)

        rel_ranks = logit_ranks(rel_logit, rel)
        self.rel_ranks = torch.cat([self.rel_ranks, rel_ranks], dim=-1)

    def compute(self):
        values = {}
        values["r_mrr"] = mrr_value(self.rel_ranks)
        for k in (1, 3, 10):
            values[f"r_hit@{k}"] = hit_value(self.rel_ranks, k)
        values["e_mrr"] = mrr_value(self.ent_ranks)
        for k in (1, 3, 10):
            values[f"e_hit@{k}"] = hit_value(self.ent_ranks, k)
        return values
