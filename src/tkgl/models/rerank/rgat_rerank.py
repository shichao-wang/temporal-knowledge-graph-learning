from typing import List

import dgl
import torch

from tkgl.modules.rgat import RelGat
from tkgl.scores import ConvTransENS

from .rerank import RerankTkgrModel, TkgrModel
from .rgcn_rerank import _build_obj_pred_graph


class RelGatRerank(RerankTkgrModel):
    def __init__(
        self,
        k: int,
        num_layers: int,
        num_channels: int,
        kernel_size: int,
        dropout: float,
        backbone: TkgrModel,
        finetune: bool,
        pretrained_backbone: str = None,
    ):
        super().__init__(backbone, finetune, pretrained_backbone)
        self.k = k
        self.rgat = RelGat(
            self.backbone.hidden_size,
            self.backbone.hidden_size,
            num_layers,
            dropout,
        )
        self.obj_score = ConvTransENS(
            self.backbone.hidden_size, num_channels, kernel_size, dropout
        )

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
        obj_pred_graph = _build_obj_pred_graph(
            self.backbone.num_ents, subj, rel, obj_logit_orig, self.k
        )
        ent_emb = backbone_outputs["ent_emb"]
        rel_emb = backbone_outputs["rel_emb"]
        node_feats = self.rgat(
            obj_pred_graph,
            ent_emb[obj_pred_graph.ndata["eid"]],
            rel_emb[obj_pred_graph.edata["rid"]],
        )
        obj_logit = self.obj_score(node_feats[subj], rel_emb[rel], ent_emb)
        return {
            "obj_logit": obj_logit,
            "obj_logit_orig": obj_logit_orig,
            **backbone_outputs,
        }

        pass
