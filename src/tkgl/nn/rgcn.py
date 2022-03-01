import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as tf
from dgl.udf import EdgeBatch, NodeBatch


class RGCNLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(RGCNLayer, self).__init__()
        self._linear = nn.Linear(2 * input_size, hidden_size, bias=False)

    def forward(self, graph: dgl.DGLGraph):
        def message_func(edges: EdgeBatch):
            h = torch.cat([edges.src["h"], edges.data["h"]], dim=-1)
            msg = self._linear(h) * edges.data["norm"]
            return {"msg": msg}

        def apply_func(nodes: NodeBatch):
            h = nodes.data["h"]
            return {"h": tf.rrelu(h)}

        graph.update_all(message_func, fn.sum(msg="msg", out="h"), apply_func)


class RGCN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self._layers = nn.ModuleList([RGCNLayer(input_size, hidden_size)])
        for _ in range(1, num_layers):
            self._layers.append(RGCNLayer(hidden_size, hidden_size))

    def forward(
        self,
        graph: dgl.DGLGraph,
        ent_embeds: torch.Tensor,
        rel_embeds: torch.Tensor,
    ):
        graph.ndata["h"] = ent_embeds[graph.ndata["ent_id"]]
        graph.edata["h"] = rel_embeds[graph.edata["rel_id"]]

        for layer in self._layers:
            layer(graph)

        return graph.ndata["h"]
