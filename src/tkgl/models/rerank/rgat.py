from typing import List

import dgl
import torch
from tallow.nn import forwards

from tkgl.modules.convtranse import ConvTransE
from tkgl.modules.mrgcn import MultiRelGraphConv

from .rerank import RerankTkgrModel
from .rgcn_rerank import build_candidate_subgraph


class RelGatRerank(RerankTkgrModel):
    def __init__(
        self,
        num_ents: int,
        num_rels: int,
        pretrained_backbone: str,
        k: int,
        rgcn_num_heads: int,
        rgcn_num_layers: int,
        convtranse_num_channels: int,
        convtranse_kernel_size: int,
        dropout: float,
        finetune: bool,
        config_path: str = None,
    ):
        super().__init__(
            num_ents, num_rels, pretrained_backbone, finetune, config_path
        )
        self.k = k
        self.rgat = MultiRelGraphConv(
            self.hidden_size,
            self.hidden_size,
            rgcn_num_heads,
            rgcn_num_layers,
            dropout,
        )
        self.convtranse = ConvTransE(
            self.backbone.hidden_size,
            2,
            convtranse_num_channels,
            convtranse_kernel_size,
            dropout,
        )

    def forward(
        self,
        hist_graphs: List[dgl.DGLGraph],
        subj: torch.Tensor,
        rel: torch.Tensor,
        obj: torch.Tensor,
    ):
        with torch.set_grad_enabled(self.finetune):
            backbone_outputs = forwards.module_forward(
                self.backbone,
                hist_graphs=hist_graphs,
                subj=subj,
                rel=rel,
                obj=obj,
            )

        obj_logit_orig = dict.pop(backbone_outputs, "obj_logit")
        candidate_subgraph = build_candidate_subgraph(
            self.backbone.num_ents, subj, rel, obj_logit_orig, self.k
        )
        ent_emb = backbone_outputs["ent_emb"]
        rel_emb = backbone_outputs["rel_emb"]
        node_feats = self.rgat(
            candidate_subgraph,
            ent_emb[candidate_subgraph.ndata["eid"]],
            rel_emb[candidate_subgraph.edata["rid"]],
        )
        enhanced_ent_emb = node_feats[
            torch.argsort(candidate_subgraph.ndata["eid"])
        ]
        pred_inp = torch.stack([enhanced_ent_emb[subj], rel_emb[rel]], dim=1)
        obj_logit = self.convtranse(pred_inp) @ ent_emb.t()
        return {
            "obj_logit": obj_logit,
            "obj_logit_orig": obj_logit_orig,
            **backbone_outputs,
        }
