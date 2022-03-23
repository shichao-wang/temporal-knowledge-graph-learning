import dgl
import torch
from dgl.udf import EdgeBatch, NodeBatch


class RelGat(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        layer_class = RelGatLayer
        layers = [layer_class(input_size, hidden_size,  dropout)]
        for _ in range(1, num_layers):
            layers.append(
                layer_class(hidden_size, hidden_size,  dropout)
            )
        self._layers = torch.nn.ModuleList(layers)
        self._dp = torch.nn.Dropout(dropout)

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ):
        nfeats = node_feats
        for layer in self._layers:
            nfeats = layer(graph, nfeats, edge_feats)
        return nfeats


class RelGatLayer(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self._linear1 = torch.nn.Linear(input_size, hidden_size)
        self._linear2 = torch.nn.Linear(input_size, hidden_size)
        self._linear3 = torch.nn.Linear(input_size, hidden_size)
        self._dropout = torch.nn.Dropout(dropout)
        self._activation = torch.nn.RReLU()

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ):
        def message_fn(edges: EdgeBatch):
            msg = self._linear1(edges.src["h"] + edges.data["h"])
            return {"msg": msg}

        def reduce_fn(nodes: NodeBatch):
            neigh_msg = nodes.mailbox["msg"]
            scores = torch.unsqueeze(nodes.data["h"], dim=1) @ torch.transpose(
                neigh_msg, -1, -2
            )
            weights = torch.softmax(scores, dim=-1)
            neigh_agg = torch.mean(weights @ neigh_msg, dim=1)
            return {"msg": neigh_agg}

        node_feats = self._dropout(node_feats)
        edge_feats = self._dropout(edge_feats)
        with graph.local_scope():
            graph.ndata["h"] = node_feats
            graph.edata["h"] = edge_feats
            graph.update_all(message_fn, reduce_fn)
            neigh_msg = graph.ndata["msg"]

        self_msg = self._linear2(node_feats)

        isolate_nids = graph.in_degrees() == 0
        iso_msg = self._linear3(node_feats)
        self_msg[isolate_nids] = iso_msg[isolate_nids]
        return self._activation(self_msg + neigh_msg)


class RelGatCatLayer(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self._linear1 = torch.nn.Linear(input_size, hidden_size)
        self._linear2 = torch.nn.Linear(input_size, hidden_size)
        self._linear3 = torch.nn.Linear(input_size, hidden_size)
        self._linear4 = torch.nn.Linear(hidden_size, num_heads)
        self._linear5 = torch.nn.Linear(input_size, num_heads)
        self._dropout = torch.nn.Dropout(dropout)
        self._activation = torch.nn.RReLU()

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ):
        def message_fn(edges: EdgeBatch):
            msg = self._linear1(edges.src["h"] + edges.data["h"])
            return {"msg": msg}

        def reduce_fn(nodes: NodeBatch):
            neigh_msg = nodes.mailbox["msg"]
            scoresT = self._linear4(neigh_msg) + self._linear5(
                torch.unsqueeze(nodes.data["h"], dim=1)
            )
            scores = torch.transpose(scoresT, -1, -2)
            weights = torch.softmax(scores, dim=-1)
            neigh_agg = torch.mean(weights @ neigh_msg, dim=1)
            return {"msg": neigh_agg}

        node_feats = self._dropout(node_feats)
        edge_feats = self._dropout(edge_feats)
        with graph.local_scope():
            graph.ndata["h"] = node_feats
            graph.edata["h"] = edge_feats
            graph.update_all(message_fn, reduce_fn)
            neigh_msg = graph.ndata["msg"]

        self_msg = self._linear2(node_feats)

        isolate_nids = graph.in_degrees() == 0
        iso_msg = self._linear3(node_feats)
        self_msg[isolate_nids] = iso_msg[isolate_nids]
        return self._activation(self_msg + neigh_msg)
