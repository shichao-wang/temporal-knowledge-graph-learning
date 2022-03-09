import copy
import warnings
from asyncio.log import logger
from typing import List

import dgl
import torch
from torch.nn import functional as tfn

from .regcn import OmegaRelGraphConv
from .tkgr_model import TkgrModel


class TemporalRerank(TkgrModel):
    def __init__(
        self,
        backbone: TkgrModel,
        k: int,
        num_layers: int,
        dropout: float,
        finetune: bool,
        pretrained_backbone: str = None,
    ):
        super().__init__()
        self._backbone = backbone
        self._k = k
        hid_sz = self._backbone.hidden_size
        self._rgcn = OmegaRelGraphConv(hid_sz, hid_sz, num_layers, dropout)
        self.obj_score = copy.deepcopy(self._backbone.obj_score)
        if pretrained_backbone:
            logger.info("Load backbone from %s", pretrained_backbone)
            backbone_state = torch.load(pretrained_backbone)["model"]
            self._backbone.load_state_dict(backbone_state)
        else:
            if not finetune:
                logger.warning("Plain backbone should be finetuned.")
        self._backbone.requires_grad_(finetune)

    def forward(
        self,
        hist_graphs: List[dgl.DGLGraph],
        subj: torch.Tensor,
        rel: torch.Tensor,
        obj: torch.Tensor,
    ):
        backbone_outputs = self._backbone(hist_graphs, subj, rel, obj)
        obj_logit_orig = dict.pop(backbone_outputs, "obj_logit")
        obj_pred_graph = _build_obj_pred_graph(
            self._backbone.num_ents, subj, rel, obj_logit_orig, self._k
        )
        ent_emb = backbone_outputs["ent_emb"]
        rel_emb = backbone_outputs["rel_emb"]
        node_feats = self._rgcn(
            obj_pred_graph,
            ent_emb[obj_pred_graph.ndata["eid"]],
            rel_emb[obj_pred_graph.edata["rid"]],
        )
        obj_logit = self.obj_score.forward(
            node_feats[subj], rel_emb[rel], ent_emb
        )
        return {
            "obj_logit": obj_logit,
            "obj_logit_orig": obj_logit_orig,
            **backbone_outputs,
        }

    @classmethod
    def build_criterion(cls, alpha: float, beta: float):
        return RerankLoss(alpha, beta)


class RerankLoss(torch.nn.Module):
    def __init__(self, alpha: float, beta: float):
        super().__init__()
        self._alpha = alpha
        self._beta = beta

    def forward(
        self,
        obj_logit: torch.Tensor,
        obj_logit_orig: torch.Tensor,
        obj: torch.Tensor,
        rel_logit: torch.Tensor,
        rel: torch.Tensor,
    ):
        obj_orig_loss = tfn.cross_entropy(obj_logit_orig, obj)
        rel_loss = tfn.cross_entropy(rel_logit, rel)
        obj_loss = tfn.cross_entropy(obj_logit, obj)
        orig_loss = self._alpha * obj_orig_loss + (1 - self._alpha) * rel_loss
        return self._beta * orig_loss + (1 - self._beta) * obj_loss


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
