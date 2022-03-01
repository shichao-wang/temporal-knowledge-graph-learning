from typing import List, Tuple

import dgl
import tallow as tl
import torch
from dgl.udf import EdgeBatch, NodeBatch
from torch import Tensor, nn
from torch.nn import functional as tnf

import tkgl


class EvoRGCN(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float
    ):
        super().__init__()

        self._layers = nn.ModuleList([Layer(input_size, hidden_size, dropout)])
        for _ in range(1, num_layers):
            self._layers.append(Layer(hidden_size, hidden_size, dropout))

    def forward(
        self,
        graph: dgl.DGLHeteroGraph,
        ent_embed: torch.Tensor,
        rel_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Arguments:
            graph: dgl.DGLGraph
            ent_embed: (num_nodes, input_size)
            rel_embed: (num_edges, input_size)
        """
        # with graph.local_scope():
        graph.ndata["h"] = ent_embed[graph.ndata["ent_id"]]
        graph.edata["h"] = rel_embed[graph.edata["rel_id"]]
        for layer in self._layers:
            _ = layer(graph)

        return graph.ndata["h"]


class Layer(nn.Module):
    """
    Notice:
    This implementation of RGCN Layer is not equivalent to the one decribed in the paper.
    In the paper, there is another self-evolve weight matrix(nn.Linear) for those entities do not exist in current graph.
    We migrate it with `self._loop_linear` here for simplicity.
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self._r_linear = nn.Linear(input_size, hidden_size, bias=False)
        self._sl_linear = nn.Linear(input_size, hidden_size, bias=False)
        self._el_linear = nn.Linear(input_size, hidden_size, bias=False)
        self._dp = nn.Dropout(dropout)

    def forward(
        self,
        graph: dgl.DGLHeteroGraph,
    ) -> torch.Tensor:
        """
        Arguments:
            graph: dgl's Graph object

        Return:
            output: (num_nodes, hidden_size)
        """

        def message_fn(edges: EdgeBatch):
            msg = self._r_linear(edges.src["h"] + edges.data["h"])
            return {"msg": msg}

        def apply_fn(nodes: NodeBatch):
            h = nodes.data["h"] * nodes.data["norm"]
            return {"h": h}

        self_msg = self._sl_linear(graph.ndata["h"])
        isolate_nodes = torch.masked_select(
            torch.arange(0, graph.number_of_nodes()),
            (graph.in_degrees() == 0),
        )
        iso_msg = self._el_linear(graph.ndata["h"])
        self_msg[isolate_nodes] = iso_msg[isolate_nodes]

        graph.update_all(message_fn, dgl.function.sum("msg", "h"), apply_fn)

        feats: torch.Tensor = graph.ndata["h"]
        feats = torch.rrelu(feats + self_msg)
        feats = self._dp(feats)
        return feats


class REGCN(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        hidden_size: int,
        num_layers: int,
        kernel_size: int,
        channels: int,
        dropout: float,
        norm_embeds: bool,
    ):
        super().__init__()
        self._norm_embeds = norm_embeds
        self.ent_embeds = nn.Parameter(torch.zeros(num_entities, hidden_size))
        self.rel_embeds = nn.Parameter(torch.zeros(num_relations, hidden_size))
        nn.init.normal_(self.ent_embeds)
        nn.init.xavier_uniform_(self.rel_embeds)

        self._rgcn = EvoRGCN(hidden_size, hidden_size, num_layers, dropout)
        self._linear = nn.Linear(hidden_size, hidden_size)
        self._gru = nn.GRUCell(2 * hidden_size, hidden_size)

        self._rel_decoder = tkgl.nn.ConvTransR(
            hidden_size=hidden_size,
            channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self._ent_decoder = tkgl.nn.ConvTransE(
            hidden_size=hidden_size,
            channels=channels,
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
        for graph in hist_graphs:
            # rel evolution
            # rel_ent_embeds = self.rel_embeds
            rel_ent_embeds = self._agg_rel_nodes(graph, ent_embeds)
            r_rel_embeds = torch.cat([self.rel_embeds, rel_ent_embeds], dim=-1)
            n_rel_embeds = self._gru(r_rel_embeds, rel_embeds)
            n_rel_embeds = self._origin_or_norm(n_rel_embeds)
            # entity evolution
            w_ent_embeds = self._rgcn(graph, ent_embeds, n_rel_embeds)
            w_ent_embeds = self._origin_or_norm(w_ent_embeds)
            u = torch.sigmoid(self._linear(ent_embeds))
            n_ent_embeds = ent_embeds + u * (w_ent_embeds - ent_embeds)

            ent_embeds = n_ent_embeds
            rel_embeds = n_rel_embeds

        rel_logit = self._rel_decoder(ent_embeds, rel_embeds, subj, obj)
        obj_logit = self._ent_decoder(ent_embeds, rel_embeds, subj, rel)

        return {"obj_logit": obj_logit, "rel_logit": rel_logit}

    def _origin_or_norm(self, tensor: Tensor):
        if self._norm_embeds:
            return tnf.normalize(tensor)
        return tensor

    def _agg_rel_nodes(self, graph: dgl.DGLGraph, ent_embeds: torch.Tensor):

        rel_ent_embeds = torch.zeros_like(self.rel_embeds)
        rel_ent_embed_list = torch.split_with_sizes(
            ent_embeds[graph.rel_node_ids], graph.rel_len
        )
        for rel, ent_embed in enumerate(rel_ent_embed_list):
            if ent_embed.size(0) != 0:
                rel_ent_embeds[rel] = torch.mean(ent_embed, dim=0)

        return rel_ent_embeds
