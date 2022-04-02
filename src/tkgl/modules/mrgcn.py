from typing import Callable

import dgl
import torch
from dgl.udf import EdgeBatch, NodeBatch


class MultiRelGraphLayer(torch.nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_heads: int, dropout: float
    ):
        super().__init__()
        self.trip_linear = torch.nn.Linear(3 * input_size, hidden_size)
        self.trip_score_linear = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, num_heads), torch.nn.LeakyReLU()
        )
        self._dropout = torch.nn.Dropout(dropout)
        self._activation = torch.nn.RReLU()
        self.selflinear = torch.nn.Linear(input_size, hidden_size)

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ):
        def message_fn(edges: EdgeBatch):
            trip_inp = torch.cat(
                [edges.data["h"], edges.src["h"], edges.dst["h"]], dim=-1
            )
            trip_hid = self.trip_linear(trip_inp)
            return {"trip_hid": trip_hid}

        # def reduce_fn(nodes: NodeBatch):
        #     """
        #     (N, K, H)
        #     """
        #     weights = torch.softmax(nodes.mailbox["trip_score"], dim=1)
        #     in_msg = weights.transpose(-1, -2) @ nodes.mailbox["trip_msg"]
        #     return {"in_msg": torch.mean(in_msg, dim=1)}

        with graph.local_scope():
            graph.ndata["h"] = node_feats
            graph.edata["h"] = edge_feats

            graph.apply_edges(message_fn)
            trip_score = self.trip_score_linear(graph.edata["trip_hid"])
            trip_head_weights = dgl.ops.edge_softmax(
                graph, torch.unsqueeze(trip_score, dim=-1)
            )
            trip_weight = torch.mean(trip_head_weights, dim=1)
            graph.edata["trip_hid_w"] = trip_weight * graph.edata["trip_hid"]

            graph.update_all(
                dgl.function.copy_edge("trip_hid_w", "trip_hid_w"),
                dgl.function.sum("trip_hid_w", "trip_msg"),
            )
            in_msg = graph.ndata["trip_msg"]

        self_msg = self.selflinear(node_feats)
        node_feats = in_msg + self_msg
        node_feats = self._activation(node_feats)
        return self._dropout(node_feats)


class MultiRelGraphConv(torch.nn.Module):
    def __init__(
        self,
        input_sizse: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        nlayer = MultiRelGraphLayer
        _nlayers = [nlayer(input_sizse, hidden_size, num_heads, dropout)]
        for _ in range(1, num_layers):
            _nlayers.append(
                nlayer(hidden_size, hidden_size, num_heads, dropout)
            )
        self.nlayers = torch.nn.ModuleList(_nlayers)

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
        for layer in self.nlayers:
            node_feats = layer(graph, node_feats, edge_feats)
        return node_feats
