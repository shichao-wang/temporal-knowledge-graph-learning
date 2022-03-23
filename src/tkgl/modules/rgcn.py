from typing import Callable

import dgl
import torch
from dgl.nn.pytorch import RelGraphConv
from dgl.udf import EdgeBatch


class RGCN(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_rels: int,
        num_layers: int,
        dropout: float = 0.0,
        activation: Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        super().__init__()
        self._num_rels = num_rels
        self._dropout = torch.nn.Dropout(dropout)
        self._activation = activation

        self._inp_layer = RelGraphConv(
            input_size,
            hidden_size,
            num_rels,
            self_loop=True,
        )
        self._hid_layers = torch.nn.ModuleList(
            [
                RelGraphConv(
                    hidden_size,
                    hidden_size,
                    num_rels,
                    self_loop=True,
                )
                for _ in range(1, num_layers)
            ]
        )

    def forward(self, graph: dgl.DGLGraph, node_feats: torch.Tensor):
        norm = self._compute_norm_coeff(graph).unsqueeze(dim=-1)
        nfeats = self._dropout(node_feats)
        nfeats = self._inp_layer(node_feats)
        for layer in self._hid_layers:
            if self._activation:
                nfeats = self._activation(nfeats)
            nfeats = self._dropout(nfeats)
            nfeats = layer(graph, nfeats, graph.edata["rid"], norm)
        return node_feats

    def _compute_norm_coeff(self, graph: dgl.DGLGraph) -> torch.Tensor:
        node_rel_counts = torch.zeros(
            graph.num_nodes(),
            self._num_rels,
            dtype=torch.float,
            device=graph.device,
        )
        _, dst, eids = graph.edges("all")
        rel_ids = graph.edata["rid"][eids]
        node_rel_counts[dst, rel_ids] += 1.0
        with graph.local_scope():
            graph.ndata["rel_counts"] = node_rel_counts
            graph.apply_edges(_retrieve_norm)
            norm = graph.edata["norm"]
        return norm


def _retrieve_norm(edges: EdgeBatch):
    num_in_degrees = edges.dst["rel_counts"][
        torch.arange(edges.batch_size()), edges.data["rid"]
    ]
    num_in_degrees = torch.clamp(num_in_degrees, 1)
    return {"norm": 1 / num_in_degrees}
