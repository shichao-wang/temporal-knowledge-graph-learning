from typing import List

import dgl
import torch
from torch import nn
from torch.nn import functional as tfn

from .regcn import ConvTransE, OmegaRelGraphConv


class Tconv(nn.Module):
    def __init__(
        self,
        num_ents: int,
        num_rels: int,
        hidden_size: int,
        num_kernels: int,
        num_layers: int,
        channels: int,
        kernel_size: int,
        dropout: float,
        norm_embeds: bool,
    ):
        super().__init__()
        self._norm_embeds = norm_embeds
        self.ent_embeds = nn.Parameter(torch.zeros(num_ents, hidden_size))
        self.rel_embeds = nn.Parameter(torch.zeros(num_rels, hidden_size))
        self.mmt_embed = nn.Parameter(torch.empty(1, hidden_size))
        nn.init.normal_(self.ent_embeds)
        nn.init.xavier_uniform_(self.rel_embeds)
        nn.init.xavier_uniform_(self.mmt_embed)

        self._rgcn = OmegaRelGraphConv(
            hidden_size, hidden_size, num_layers, dropout
        )
        self._linear = nn.Linear(hidden_size, hidden_size)
        self._gru1 = nn.GRUCell(2 * hidden_size, hidden_size)
        self._linear1 = nn.GRUCell(hidden_size, hidden_size)
        self._linear2 = nn.GRUCell(hidden_size, hidden_size)
        self._tconv = TemporalConv(hidden_size, hidden_size, num_kernels)

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
        mmt_embed = self.mmt_embed
        for graph in hist_graphs:
            n_mmt_embed = self._linear1(mmt_embed)
            g = self._linear2(mmt_embed)
            mmt_embed = mmt_embed + g * (n_mmt_embed - mmt_embed)
            ent_embeds = self._tconv(ent_embeds, mmt_embed)
            rel_embeds = self._tconv(rel_embeds, mmt_embed)

            # rel evolution
            # rel_ent_embeds = self.rel_embeds
            rel_ent_embeds = self._agg_rel_nodes(graph, ent_embeds)
            gru_input = torch.cat([rel_ent_embeds, self.rel_embeds], dim=-1)
            n_rel_embeds = self._gru1(gru_input, rel_embeds)
            n_rel_embeds = self._origin_or_norm(n_rel_embeds)
            # entity evolution
            edge_feats = n_rel_embeds[graph.edata["rid"]]
            node_feats, _ = self._rgcn(graph, (ent_embeds, edge_feats))
            node_feats = self._origin_or_norm(node_feats)
            u = torch.sigmoid(self._linear(ent_embeds))
            n_ent_embeds = ent_embeds + u * (node_feats - ent_embeds)

            ent_embeds = n_ent_embeds
            rel_embeds = n_rel_embeds

        obj_pred = torch.stack([ent_embeds[subj], rel_embeds[rel]], dim=1)
        obj_logit = self._obj_convtranse(obj_pred) @ ent_embeds.t()
        rel_pred = torch.stack([ent_embeds[subj], ent_embeds[obj]], dim=1)
        rel_logit = self._rel_convtranse(rel_pred) @ rel_embeds.t()

        return {"obj_logit": obj_logit, "rel_logit": rel_logit}

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


class TemporalConv(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, k: int):
        super().__init__()
        self._linear = nn.Linear(input_size, hidden_size)
        self._k = k
        self._pooling = nn.Linear(
            hidden_size * k - hidden_size + k, hidden_size
        )

    def forward(self, embeds: torch.Tensor, t: torch.Tensor):
        """
        Arguments:
            embeds: (num_embeds, embed_size)
            t: (hidden_size)
        Returns:
            hiddens: (num_embeds, hidden_size)
        """
        # shape: (k, 1, hidden // k)
        v = torch.reshape(self._linear(t), (self._k, 1, -1))
        # shape: (num_embeddings, k, hidden - hidden // k + 1)
        feature = tfn.conv1d(torch.unsqueeze(embeds, dim=1), v)
        return self._pooling(feature.view(feature.size(0), -1))
