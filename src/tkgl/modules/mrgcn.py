from typing import Callable

import dgl
import torch
from dgl.udf import EdgeBatch


class MultiRelGraphLayer(torch.nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, self_loop: bool, dropout: float
    ):
        super().__init__()
        self.neigh_linear = torch.nn.Linear(2 * input_size, hidden_size)
        self._self_loop = self_loop

        self._dropout = torch.nn.Dropout(dropout)
        self._activation = torch.nn.RReLU()

        if self._self_loop:
            self.looplinear = torch.nn.Linear(input_size, hidden_size)

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ):
        def message_fn(edges: EdgeBatch):
            msg_inp = torch.cat([edges.src["h"], edges.data["h"]], dim=-1)
            msg = self.neigh_linear(msg_inp)
            return {"msg": msg}

        with graph.local_scope():
            graph.ndata["h"] = node_feats
            graph.edata["h"] = edge_feats
            # node msg
            graph.update_all(message_fn, dgl.function.mean("msg", "msg"))

            neigh_msg = graph.ndata["msg"]

        if self._self_loop:
            self_msg = self.looplinear(node_feats)
            neigh_msg = neigh_msg + self_msg
        neigh_msg = self._activation(node_feats)
        return self._dropout(neigh_msg)


class MultiRelGraphConv(torch.nn.Module):
    def __init__(
        self,
        input_sizse: int,
        hidden_size: int,
        num_layers: int,
        self_loop: bool,
        dropout: float,
    ):
        super().__init__()
        layer = MultiRelGraphLayer
        _layers = [layer(input_sizse, hidden_size, self_loop, dropout)]
        for _ in range(1, num_layers):
            _layers.append(layer(hidden_size, hidden_size, self_loop, dropout))
        self.layers = torch.nn.ModuleList(_layers)
        self.olinear = torch.nn.Linear(hidden_size * num_layers, hidden_size)

    __call__: Callable[
        [
            "MultiRelGraphConv",
            dgl.DGLGraph,
            torch.Tensor,
            torch.Tensor,
        ],
        torch.Tensor,
    ]

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ):
        neigh_feats = node_feats
        neigh_msg_list = []
        for layer in self.layers:
            neigh_feats = layer(graph, neigh_feats, edge_feats)
            neigh_msg_list.append(neigh_feats)

        multihop_neighbor = torch.cat(neigh_msg_list, dim=-1)
        return self.olinear(multihop_neighbor)
