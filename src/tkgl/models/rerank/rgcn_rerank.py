from typing import List

import dgl
import torch

from tkgl.models.regcn import OmegaRelGraphConv
from tkgl.models.tkgr_model import TkgrModel
from tkgl.scores import ConvTransENS

from .rerank import RerankTkgrModel


class RelGraphConvRerank(RerankTkgrModel):
    def __init__(
        self,
        backbone: TkgrModel,
        k: int,
        num_layers: int,
        num_channels: int,
        kernel_size: int,
        dropout: float,
        finetune: bool,
        pretrained_backbone: str = None,
    ):
        super().__init__(backbone, finetune, pretrained_backbone)
        self._k = k
        self._rgcn = OmegaRelGraphConv(
            self.hidden_size, self.hidden_size, num_layers, dropout
        )
        self.obj_score = ConvTransENS(
            self.hidden_size, num_channels, kernel_size, dropout
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
            self.backbone.num_ents, subj, rel, obj_logit_orig, self._k
        )
        ent_emb = backbone_outputs["ent_emb"]
        rel_emb = backbone_outputs["rel_emb"]
        node_feats = self._rgcn(
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


def _build_obj_pred_graph(
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
