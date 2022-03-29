from typing import List, Tuple

import dgl
import torch
from dgl.udf import EdgeBatch
from torch import nn
from torch.nn import functional as f

from tkgl.convtranse import ConvTransE
from tkgl.models.tkgr_model import TkgrModel


class OmegaRelGraphConv(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        self_loop: bool,
        dropout: float,
    ):
        super().__init__()
        layer = OmegaGraphConvLayer

        _layers = [layer(input_size, hidden_size, dropout, self_loop)]
        for _ in range(1, num_layers):
            _layers.append(layer(hidden_size, hidden_size, dropout, self_loop))
        self._layers = torch.nn.ModuleList(_layers)

    def forward(
        self,
        graph: dgl.DGLHeteroGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Arguments:
            graph: dgl.DGLGraph
            ent_embeds: (num_nodes, input_size)
            rel_embeds: (num_edges, input_size)
        """
        for layer in self._layers:
            node_feats = layer(graph, node_feats, edge_feats)
        return node_feats


class OmegaGraphConvLayer(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, dropout: float, self_loop: bool
    ):
        super().__init__()
        self.neigh_linear = nn.Linear(input_size, hidden_size, bias=False)
        self._dropout = torch.nn.Dropout(dropout)
        self._activation = torch.nn.RReLU()
        self._self_loop = self_loop

        if self._self_loop:
            self.sl_linear = nn.Linear(input_size, hidden_size, bias=False)
            self.is_linear = nn.Linear(input_size, hidden_size, bias=False)

    def forward(
        self,
        graph: dgl.DGLHeteroGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Arguments:
            graph: dgl's Graph object

        Return:
            output: (num_nodes, hidden_size)
        """

        def message_fn(edges: EdgeBatch):
            msg = self.neigh_linear(edges.src["h"] + edges.data["h"])
            return {"msg": msg}

        with graph.local_scope():
            graph.ndata["h"], graph.edata["h"] = node_feats, edge_feats
            graph.update_all(message_fn, dgl.function.mean("msg", "msg"))
            neigh_msg = graph.ndata["msg"]

        x = neigh_msg
        if self._self_loop:
            self_msg = self.sl_linear(node_feats)
            iso_msg = self.is_linear(node_feats)
            isolate_nids = graph.in_degrees() == 0
            self_msg[isolate_nids] = iso_msg[isolate_nids]
            x = x + self_msg

        x = self._activation(x)
        x = self._dropout(x)
        return x


class REGCN(TkgrModel):
    def __init__(
        self,
        num_ents: int,
        num_rels: int,
        hidden_size: int,
        dropout: float,
        norm_embeds: bool,
        rgcn_num_layers: int,
        rgcn_self_loop: bool,
        convtranse_kernel_size: int,
        convtranse_channels: int,
    ):
        super().__init__(num_ents, num_rels, hidden_size)
        self.rgcn = OmegaRelGraphConv(
            hidden_size, hidden_size, rgcn_num_layers, rgcn_self_loop, dropout
        )
        self.glinear = nn.Linear(hidden_size, hidden_size)
        self.rel_rnn = nn.GRUCell(2 * hidden_size, hidden_size)
        self.rel_convtranse = ConvTransE(
            hidden_size,
            2,
            convtranse_channels,
            convtranse_kernel_size,
            dropout,
        )
        self.obj_convtranse = ConvTransE(
            hidden_size,
            2,
            convtranse_channels,
            convtranse_kernel_size,
            dropout,
        )
        self._norm_embeds = norm_embeds

    def evolve(
        self, hist_graphs: List[dgl.DGLGraph]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ent_emb = self._origin_or_norm(self.ent_emb)
        rel_emb = self.rel_emb

        hist_ent_emb = []
        hist_rel_emb = []
        for graph in hist_graphs:
            node_feats = ent_emb[graph.ndata["eid"]]
            # relation evolution
            rel_ent_emb = self._agg_rel_nodes(graph, node_feats)
            gru_input = torch.cat([self.rel_emb, rel_ent_emb], dim=-1)
            rel_emb = self.rel_rnn(gru_input, rel_emb)
            rel_emb = self._origin_or_norm(rel_emb)
            # entity evolution
            edge_feats = rel_emb[graph.edata["rid"]]
            neigh_feats = self.rgcn(graph, node_feats, edge_feats)
            neigh_feats = self._origin_or_norm(neigh_feats)
            cur_ent_emb = neigh_feats[torch.argsort(graph.ndata["eid"])]
            u = torch.sigmoid(self.glinear(cur_ent_emb))
            ent_emb = u * cur_ent_emb + (1 - u) * ent_emb
            ent_emb = self._origin_or_norm(ent_emb)

            hist_ent_emb.append(ent_emb)
            hist_rel_emb.append(rel_emb)

        return torch.stack(hist_ent_emb), torch.stack(hist_rel_emb)

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
        hist_ent_emb, hist_rel_emb = self.evolve(hist_graphs)
        ent_emb = hist_ent_emb[-1]
        rel_emb = hist_rel_emb[-1]

        subj_emb = ent_emb[subj]
        obj_inp = torch.stack([subj_emb, rel_emb[rel]], dim=1)
        obj_logit = self.obj_convtranse(obj_inp) @ ent_emb.t()
        rel_inp = torch.stack([subj_emb, ent_emb[obj]], dim=1)
        rel_logit = self.obj_convtranse(rel_inp) @ ent_emb.t()
        return {"obj_logit": obj_logit, "rel_logit": rel_logit}

    def _origin_or_norm(self, tensor: torch.Tensor):
        if self._norm_embeds:
            return f.normalize(tensor)
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
            self.rel_emb.size(0), node_feats.size(0), dtype=torch.bool
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
