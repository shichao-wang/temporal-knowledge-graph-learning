from typing import List

import dgl
import molurus
import torch
from tallow.nn import forwards

from tkgl.models.regcn import OmegaRelGraphConv
from tkgl.modules.convtranse import ConvTransE
from tkgl.modules.mrgcn import MultiRelGraphConv

from .rerank import RerankTkgrModel


class RelGraphConvRerank(RerankTkgrModel):
    def __init__(
        self,
        num_ents: int,
        num_rels: int,
        pretrained_backbone: str,
        finetune: bool,
        k: int,
        rgcn_num_layers: int,
        rgcn_self_loop: bool,
        convtranse_num_channels: int,
        convtranse_kernel_size: int,
        dropout: float,
        config_path: str = None,
    ):
        super().__init__(
            num_ents, num_rels, pretrained_backbone, finetune, config_path
        )
        self.k = k
        self.rgcn = OmegaRelGraphConv(
            self.hidden_size,
            self.hidden_size,
            rgcn_num_layers,
            rgcn_self_loop,
            dropout,
        )
        self.obj_score = ConvTransE(
            self.hidden_size,
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
        ent_emb = backbone_outputs["hist_ent_emb"][-1]
        if "hist_rel_emb" in backbone_outputs:
            rel_emb = backbone_outputs["hist_rel_emb"][-1]
        else:
            rel_emb = self.backbone.rel_emb

        node_feats = self.rgcn(
            candidate_subgraph,
            ent_emb[candidate_subgraph.ndata["eid"]],
            rel_emb[candidate_subgraph.edata["rid"]],
        )
        enhanced_ent_emb = node_feats[
            torch.argsort(candidate_subgraph.ndata["eid"])
        ]
        pred_inp = torch.stack([enhanced_ent_emb[subj], rel_emb[rel]], dim=1)
        obj_logit = self.obj_score(pred_inp) @ ent_emb.t()
        return {
            "obj_logit": obj_logit,
            "obj_logit_orig": obj_logit_orig,
            **backbone_outputs,
        }


def build_candidate_subgraph(
    num_nodes: int,
    subj: torch.Tensor,
    rel: torch.Tensor,
    obj_logit: torch.Tensor,
    k: int,
):
    _, topk_obj = torch.topk(obj_logit, k=k)
    total_subj = torch.repeat_interleave(subj, k)
    total_rel = torch.repeat_interleave(rel, k)
    total_obj = topk_obj.view(-1)
    pred_graph = dgl.graph(
        (total_subj, total_obj),
        num_nodes=num_nodes,
        device=obj_logit.device,
    )
    pred_graph.ndata["eid"] = torch.arange(num_nodes, device=pred_graph.device)
    pred_graph.edata["rid"] = total_rel
    return pred_graph
