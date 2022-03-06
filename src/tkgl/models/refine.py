from typing import List

import dgl
import torch
from tallow.nn import forwards
from torch import nn
from torch.nn import functional as tfn

from tkgl.modules import RGCN

from .criterions import JointLoss
from .regcn import ConvTransE, OmegaRelGraphConv


class RefineLoss(torch.nn.Module):
    def __init__(self, alpha: float, beta: float):
        super().__init__()
        self._join_loss = JointLoss(alpha)
        self._beta = beta

    def forward(
        self,
        obj_logit: torch.Tensor,
        obj: torch.Tensor,
        rel_logit: torch.Tensor,
        rel: torch.Tensor,
        obj_logit_orig: torch.Tensor,
    ):
        v1 = self._join_loss(obj_logit, obj, rel_logit, rel)
        v2 = tfn.cross_entropy(obj_logit_orig, obj)
        return self._beta * v1 + (1 - self._beta) * v2


class Refine(torch.nn.Module):
    def __init__(
        self,
        num_ents: int,
        num_rels: int,
        hidden_size: int,
        num_layers: int,
        channels: int,
        kernel_size: int,
        dropout: float,
        norm_embeds: bool,
        num_heads: int,
        k: int,
    ):
        super().__init__()
        self._k = k
        self._norm_embeds = norm_embeds

        self.ent_embeds = torch.nn.Parameter(torch.zeros(num_ents, hidden_size))
        self.rel_embeds = torch.nn.Parameter(torch.zeros(num_rels, hidden_size))
        torch.nn.init.normal_(self.ent_embeds)
        torch.nn.init.xavier_uniform_(self.rel_embeds)

        self._omega_rgcn = OmegaRelGraphConv(
            hidden_size, hidden_size, num_layers, dropout
        )
        self._linear = nn.Linear(hidden_size, hidden_size)
        self._gru = nn.GRUCell(2 * hidden_size, hidden_size)

        self._rel_convtranse = ConvTransE(
            hidden_size=hidden_size,
            in_channels=2,
            out_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self._obj_convtranse = ConvTransE(
            hidden_size=hidden_size,
            in_channels=2,
            out_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self._linear2 = torch.nn.Linear(2 * hidden_size, hidden_size)
        self._multihead = torch.nn.MultiheadAttention(
            hidden_size, num_heads, dropout
        )

    def forward(
        self,
        hist_graphs: List[dgl.DGLGraph],
        subj: torch.Tensor,
        rel: torch.Tensor,
        obj: torch.Tensor,
    ):
        """

        Arguments:
            snapshot: [his_len]
            triplets: (num_triplets, 3)

        Returns:
            logits: (num_triplets, num_entities)
        """
        ent_embeds = self._origin_or_norm(self.ent_embeds)
        rel_embeds = self._origin_or_norm(self.rel_embeds)
        for graph in hist_graphs:
            ent_embeds, rel_embeds = self._regcn_evolve(
                graph, ent_embeds, rel_embeds
            )

        obj_pred = torch.stack([ent_embeds[subj], rel_embeds[rel]], dim=1)
        obj_logit = self._obj_convtranse(obj_pred) @ ent_embeds.t()
        _, topk_obj = torch.topk(obj_logit, self._k)
        q = self._linear2(
            torch.cat([ent_embeds[subj], rel_embeds[rel]], -1).unsqueeze(1)
        )
        pred_obj_embeds = ent_embeds[topk_obj]
        rerank_logit, _ = forwards.mh_attention_forward(
            self._multihead, q, pred_obj_embeds, pred_obj_embeds
        )
        rerank_logit = rerank_logit.squeeze(dim=1)
        obj_logit = rerank_logit @ ent_embeds.t()
        rel_pred = torch.stack([ent_embeds[subj], ent_embeds[obj]], dim=1)
        rel_logit = self._rel_convtranse(rel_pred) @ rel_embeds.t()

        return {
            "obj_logit": obj_logit,
            "rel_logit": rel_logit,
        }

    def _regcn_evolve(
        self,
        graph: dgl.DGLGraph,
        ent_embeds: torch.Tensor,
        rel_embeds: torch.Tensor,
    ):
        # rel evolution
        rel_ent_embeds = self._agg_rel_nodes(graph, ent_embeds)
        gru_input = torch.cat([rel_ent_embeds, self.rel_embeds], dim=-1)
        n_rel_embeds = self._gru(gru_input, rel_embeds)
        n_rel_embeds = self._origin_or_norm(n_rel_embeds)
        # entity evolution
        edge_feats = n_rel_embeds[graph.edata["rid"]]
        node_feats = ent_embeds[graph.ndata["eid"]]
        node_feats, _ = self._omega_rgcn(graph, (node_feats, edge_feats))
        node_feats = self._origin_or_norm(node_feats)
        u = torch.sigmoid(self._linear(ent_embeds))
        n_ent_embeds = ent_embeds + u * (node_feats - ent_embeds)
        return n_ent_embeds, n_rel_embeds

    def _build_obj_pred_graph(
        self,
        subj: torch.Tensor,
        rel: torch.Tensor,
        obj_logit: torch.Tensor,
        k: int,
    ):
        _, topk_obj = torch.topk(obj_logit, k=k)
        num_nodes = self.ent_embeds.size(0)
        total_subj = torch.repeat_interleave(subj, k)
        total_rel = torch.repeat_interleave(rel, k)
        total_obj = topk_obj.view(-1)
        pred_graph = dgl.graph(
            (total_subj, total_obj),
            num_nodes=num_nodes,
            device=obj_logit.device,
        )
        pred_graph.ndata["eid"] = torch.arange(
            num_nodes, device=obj_logit.device
        )
        pred_graph.edata["rid"] = total_rel
        return pred_graph

    def _origin_or_norm(self, tensor: torch.Tensor):
        if self._norm_embeds:
            return tfn.normalize(tensor)
        return tensor

    def _agg_rel_nodes(self, graph: dgl.DGLGraph, node_feats: torch.Tensor):
        """
        Arguments:
            nfeats: (num_nodes, hidden_size)
        Return:
            (num_rels, hidden_size)
        """
        # (num_rels, num_nodes)
        rel_node_mask = node_feats.new_zeros(
            self.rel_embeds.size(0), node_feats.size(0), dtype=torch.bool
        )
        src, dst, eids = graph.edges("all")
        rel_ids = graph.edata["rid"][eids]
        rel_node_mask[rel_ids, src] = True
        rel_node_mask[rel_ids, dst] = True

        node_ids = torch.nonzero(rel_node_mask)[:, 1]
        rel_embeds = dgl.ops.segment_reduce(
            rel_node_mask.sum(dim=1), node_feats[node_ids], "mean"
        )
        return torch.nan_to_num(rel_embeds, 0)
