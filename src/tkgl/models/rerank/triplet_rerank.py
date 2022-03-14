import logging
from typing import List

import dgl
import torch
from tallow.nn import forwards

from tkgl.models.tkgr_model import JointLoss, TkgrModel

from .rerank import RerankTkgrModel

logger = logging.getLogger(__name__)


class TripletRerank(RerankTkgrModel):
    def __init__(
        self,
        backbone: TkgrModel,
        k: int,
        num_heads: int,
        dropout: float,
        finetune: bool,
        pretrained_backbone: str = None,
    ):
        super().__init__(backbone, finetune, pretrained_backbone)
        self._k = k
        self._linear1 = torch.nn.Linear(3 * self.hidden_size, self.hidden_size)
        self._linear2 = torch.nn.Linear(2 * self.hidden_size, self.hidden_size)
        self._mh1 = torch.nn.MultiheadAttention(
            self.hidden_size, num_heads, dropout
        )
        self._mh2 = torch.nn.MultiheadAttention(
            self.hidden_size, num_heads, dropout
        )
        self._linear3 = torch.nn.Linear(
            self.hidden_size, self.backbone.num_ents
        )
        self._gate_add = GateNet(self.backbone.num_ents)

    def forward(
        self,
        hist_graphs: List[dgl.DGLGraph],
        subj: torch.Tensor,
        rel: torch.Tensor,
        obj: torch.Tensor,
    ):
        with torch.set_grad_enabled(self.finetune):
            backbone_outputs = self.backbone(hist_graphs, subj, rel, obj)
        obj_logit_orig = dict.pop(backbone_outputs, "obj_logit")
        # rel_logit_orig = dict.pop(backbone_outputs, "rel_logit")
        _, topk_obj = torch.topk(obj_logit_orig, k=self._k)
        total_subj = torch.repeat_interleave(subj, self._k)
        total_rel = torch.repeat_interleave(rel, self._k)
        total_trips = torch.cat(
            [
                backbone_outputs["ent_emb"][total_subj],
                backbone_outputs["rel_emb"][total_rel],
                backbone_outputs["ent_emb"][topk_obj.view(-1)],
            ],
            dim=-1,
        )
        ctx_trips = self._linear1(total_trips)

        # (N, H * 2)
        query_inp = torch.cat(
            [
                backbone_outputs["ent_emb"][subj],
                backbone_outputs["rel_emb"][rel],
            ],
            dim=-1,
        )

        # (N, 1, H), (N, 1, C)
        query_trips, _ = forwards.mh_attention_forward(
            self._mh1,
            torch.unsqueeze(self._linear2(query_inp), dim=1),
            ctx_trips,
            ctx_trips,
        )
        rerank_trips, _ = forwards.mh_attention_forward(
            self._mh2, query_trips, ctx_trips, ctx_trips
        )
        obj_logit = self._gate_add(
            obj_logit_orig, self._linear3(rerank_trips.squeeze(dim=1))
        )

        return {
            "obj_logit": obj_logit,
            # "obj_logit_orig": obj_logit_orig,
            **backbone_outputs,
        }

    @classmethod
    def build_criterion(cls, alpha: float, beta: float):
        return JointLoss(alpha)


class GateNet(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self._linear = torch.nn.Linear(2 * hidden_size, 1)

    def forward(self, orig_inp: torch.Tensor, add_inp: torch.Tensor):
        x = torch.cat([orig_inp, add_inp], dim=-1)
        gate = torch.sigmoid(self._linear(x))
        return gate * orig_inp + (1 - gate) * add_inp
