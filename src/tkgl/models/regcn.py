from typing import List, Tuple

import dgl
import torch
from dgl.udf import EdgeBatch
from torch import nn
from torch.nn import functional as tnf

from tkgl.models.tkgr_model import TkgrModel
from tkgl.scores import ConvTransENS, ConvTransERS


class OmegaRelGraphConv(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float
    ):
        super().__init__()
        self._inp_layer = self.Layer(input_size, hidden_size)

        _layers = [
            self.Layer(hidden_size, hidden_size) for _ in range(1, num_layers)
        ]
        self._layers = torch.nn.ModuleList(_layers)
        self._dp = torch.nn.Dropout(dropout)

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
        nfeats = self._inp_layer(graph, node_feats, edge_feats)
        for layer in self._layers:
            nfeats, efeats = self._dp(nfeats), self._dp(edge_feats)
            nfeats = layer(graph, nfeats, efeats)

        return nfeats

    class Layer(nn.Module):
        """
        Notice:
        This implementation of RGCN Layer is not equivalent to the one decribed in the paper.
        In the paper, there is another self-evolve weight matrix(nn.Linear) for those entities do not exist in current graph.
        We migrate it with `self._loop_linear` here for simplicity.
        """

        def __init__(self, input_size: int, hidden_size: int):
            super().__init__()
            self._linear1 = nn.Linear(input_size, hidden_size, bias=False)
            self._linear2 = nn.Linear(input_size, hidden_size, bias=False)
            self._linear3 = nn.Linear(input_size, hidden_size, bias=False)

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
                msg = self._linear1(edges.src["h"] + edges.data["h"])
                return {"msg": msg}

            with graph.local_scope():
                graph.ndata["h"], graph.edata["h"] = node_feats, edge_feats
                graph.update_all(message_fn, dgl.function.mean("msg", "msg"))
                neigh_msg = graph.ndata["msg"]

            self_msg = self._linear2(node_feats)
            isolate_nids = torch.masked_select(
                torch.arange(0, graph.number_of_nodes()),
                (graph.in_degrees() == 0),
            )
            iso_msg = self._linear3(node_feats)
            self_msg[isolate_nids] = iso_msg[isolate_nids]

            node_feats = torch.rrelu(neigh_msg + self_msg)
            return node_feats


class REGCN(TkgrModel):
    def __init__(
        self,
        num_ents: int,
        num_rels: int,
        hidden_size: int,
        num_layers: int,
        kernel_size: int,
        channels: int,
        dropout: float,
        norm_embeds: bool,
    ):
        super().__init__()
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.hidden_size = hidden_size

        self._norm_embeds = norm_embeds
        self.ent_embeds = nn.Parameter(torch.zeros(num_ents, hidden_size))
        self.rel_embeds = nn.Parameter(torch.zeros(num_rels, hidden_size))
        nn.init.normal_(self.ent_embeds)
        nn.init.xavier_uniform_(self.rel_embeds)

        self._rgcn = OmegaRelGraphConv(
            hidden_size, hidden_size, num_layers, dropout
        )
        self._linear = nn.Linear(hidden_size, hidden_size)
        self._gru = nn.GRUCell(2 * hidden_size, hidden_size)

        self.rel_score = ConvTransERS(
            hidden_size=hidden_size,
            num_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.obj_score = ConvTransENS(
            hidden_size=hidden_size,
            num_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def evolve(
        self, hist_graphs: List[dgl.DGLGraph]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ent_embeds = self._origin_or_norm(self.ent_embeds)
        rel_embeds = self._origin_or_norm(self.rel_embeds)
        for graph in hist_graphs:
            # rel evolution
            # rel_ent_embeds = self.rel_embeds
            rel_ent_embeds = self._agg_rel_nodes(graph, ent_embeds)
            gru_input = torch.cat([rel_ent_embeds, self.rel_embeds], dim=-1)
            n_rel_embeds = self._gru(gru_input, rel_embeds)
            n_rel_embeds = self._origin_or_norm(n_rel_embeds)
            # entity evolution
            edge_feats = n_rel_embeds[graph.edata["rid"]]
            node_feats = self._rgcn(graph, ent_embeds, edge_feats)
            node_feats = self._origin_or_norm(node_feats)
            u = torch.sigmoid(self._linear(ent_embeds))
            n_ent_embeds = ent_embeds + u * (node_feats - ent_embeds)

            ent_embeds = n_ent_embeds
            rel_embeds = n_rel_embeds

        return ent_embeds, rel_embeds

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
        ent_embeds, rel_embeds = self.evolve(hist_graphs)
        subj_embeds = ent_embeds[subj]
        obj_logit = self.obj_score(subj_embeds, rel_embeds[rel], ent_embeds)
        rel_logit = self.rel_score(subj_embeds, ent_embeds[obj], rel_embeds)

        return {
            "obj_logit": obj_logit,
            "rel_logit": rel_logit,
            "ent_emb": ent_embeds,
            "rel_emb": rel_embeds,
        }

    def _origin_or_norm(self, tensor: torch.Tensor):
        if self._norm_embeds:
            return tnf.normalize(tensor)
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
