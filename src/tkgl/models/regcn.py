from typing import List, Tuple

import dgl
import torch
import torch_helpers
from dgl.udf import EdgeBatch, NodeBatch
from torch import nn
from torch.nn import functional as tf
from torch_helpers.nn.embeddings import Embedding


class RGCN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()

        self._layers = nn.ModuleList([RGCN.Layer(input_size, hidden_size)])
        for _ in range(1, num_layers):
            self._layers.append(RGCN.Layer(hidden_size, hidden_size))

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
        with graph.local_scope():
            graph.ndata["h"] = ent_embed
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

        def __init__(self, input_size: int, hidden_size: int):
            super().__init__()
            self._rel_linear = nn.Linear(input_size, hidden_size, bias=False)
            self._loop_linear = nn.Linear(input_size, hidden_size, bias=False)
            self._rrelu = nn.RReLU()

        def forward(
            self,
            graph: dgl.DGLHeteroGraph,
            features: Tuple[torch.Tensor, torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Arguments:
                graph: dgl's Graph object
                nodes: nodes embedding (num_nodes, hidden_size)
                edges: edges embedding (num_edges, hidden_size)

            Return:
                output: (num_nodes, hidden_size)
            """
            if features:
                nodes, edges = features
                graph.srcdata["h"] = nodes
                graph.edata["h"] = edges

            self_msg = self._loop_linear(graph.ndata["h"])

            graph.update_all(
                self.message_fn, dgl.function.sum("msg", "h"), self.apply_fn
            )
            nodes = graph.ndata["h"] + self_msg
            return self._rrelu(nodes)

        def message_fn(self, edges: EdgeBatch):
            rel = edges.data["h"]
            ent = edges.src["h"]

            msg = self._rel_linear(rel + ent)

            return {"msg": msg}

        def apply_fn(self, nodes: NodeBatch):
            return {"h": nodes.data["h"] * nodes.data["norm"]}


class EvolutionUnit(nn.Module):
    """
    Arguments:
        num_layers: \omega
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self._rgcn = RGCN(input_size, hidden_size, num_layers)
        self._linear = nn.Linear(hidden_size, hidden_size)
        self._gru = nn.GRUCell(2 * hidden_size, hidden_size)

    def forward(
        self,
        graph: dgl.DGLGraph,
        ent_embed: torch.Tensor,
        rel_embed: torch.Tensor,
    ):
        """
        Arguments:
            graph: dgl.Graph
                The knowledge graph at time t
            node_embedding: (num_total_nodes, hidden_size)
                H_{t-1}
            edge_embedding: (num_edges, hidden_size)
                R_{t-1}
        Returns:

        """
        # relaltion evolution
        rel_ent_embed = torch.zeros_like(rel_embed)
        for rel, rel_ent_ids in graph.rel_to_ent.items():
            embed = torch.mean(ent_embed[rel_ent_ids], dim=0)
            rel_ent_embed[rel] = embed

        r_rel_embed = torch.cat([rel_embed, rel_ent_embed], dim=-1)
        n_rel_embed = tf.normalize(self._gru(r_rel_embed, rel_embed))
        # entity evolution
        w_ent_embed = self._rgcn(graph, ent_embed, n_rel_embed)
        w_ent_embed = tf.normalize(w_ent_embed)
        u = torch.sigmoid(self._linear(ent_embed))
        n_ent_embed = ent_embed + u * (w_ent_embed - ent_embed)

        return n_ent_embed, n_rel_embed


class RecurrentRGCN(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        hidden_size: int,
        num_layers: int,
    ):
        super().__init__()
        ent_weight = torch_helpers.nn.random_init_embedding_weight(
            num_entities, hidden_size
        )
        rel_weight = torch_helpers.nn.random_init_embedding_weight(
            num_relations, hidden_size
        )
        self._ent_embed = nn.Parameter(ent_weight)
        self._rel_embed = nn.Parameter(rel_weight)
        self._evolution = EvolutionUnit(hidden_size, hidden_size, num_layers)

    def forward(self, snapshots: List[dgl.DGLGraph]):
        """
        Arguments:
            snapshot: [his_len]
            entity_embedding: (num_entities, input_size)
            relation_embedding: (num_relations, input_size)

        Returns:
            entity: (his_len, num_nodes, hidden_size)
                The node embeddings for each snapshot
            relations: (his_len, num_relations, hidden_size)
            last_entity: (num_nodes, hidden_size)
            last_relation: (num_relations, hidden_size)
        """

        e_h, r_h = self._ent_embed, self._rel_embed
        ent_embeds, rel_embeds = [], []
        for graph in snapshots:
            e_h, r_h = self._evolution(graph, e_h, r_h)
            ent_embeds.append(e_h)
            rel_embeds.append(r_h)

        return {
            "entity": torch.stack(ent_embeds),
            "relation": torch.stack(rel_embeds),
            "last": (e_h, r_h),
        }


class RecurrentRGCNForLinkPrediction(nn.Module):
    def __init__(
        self,
        ent_embedding: Embedding,
        rel_embedding: Embedding,
        hidden_size: int,
        num_layers: int,
        kernel_size: int,
        channels: int,
        dropout: float,
    ):
        super().__init__()
        self._rrgcn = RecurrentRGCN(
            ent_embedding,
            rel_embedding,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self._lp = ConvTransR(
            hidden_size=hidden_size,
            channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(
        self,
        snapshots: List[dgl.DGLGraph],
        head: torch.Tensor,
        tail: torch.Tensor,
    ):
        """

        Arguments:
            snapshot: [his_len]
            triplets: (num_triplets, 3)

        Returns:
            logits: (num_triplets, num_entities)
        """
        evolution_outputs = self._rrgcn(snapshots)
        e_h, r_h = evolution_outputs["last"]
        outputs = self._lp(e_h, r_h, head, tail)
        return {"logits": outputs}


class ConvTransBackbone(nn.Module):
    """
    The backbone module for ConvTransE and ConvTransR.
    The Conv1d here is not a common way to process NLP features.
    """

    def __init__(
        self,
        hidden_size: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self._bn = nn.BatchNorm1d(in_channels)
        self._dp = nn.Dropout(dropout)
        self._conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        self._bn1 = nn.BatchNorm1d(out_channels)
        self._dp1 = nn.Dropout(dropout)
        self._linear = nn.Linear(hidden_size * out_channels, hidden_size)
        self._bn2 = nn.BatchNorm1d(hidden_size)
        self._dp2 = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
    ):
        """
        Arguments:
            inputs: (num_triplets, in_channels, hidden_size)
        Return:
            output: (num_triplets, hidden_size)
        """
        num_triplets = inputs.size(0)
        # (num_triplets, out_channels, embed_size)
        feature_maps = self._conv1(self._dp(self._bn(inputs)))
        _ = self._dp1(self._bn1(feature_maps).relu())
        hidden = self._linear(_.view(num_triplets, -1))
        # (num_triplets, 1, embed_size)
        embedding = self._bn2(self._dp2(hidden)).relu()
        return embedding
        # (num_triplets, embed_size)


class ConvTransE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        channels: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self._backbone = ConvTransBackbone(
            hidden_size,
            in_channels=2,
            out_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        heads: torch.Tensor,
        relations: torch.Tensor,
    ):
        """
        Arguments:
            nodes: (num_nodes, hidden_size)
            edges: (num_edges, hidden_size)
            heads: (num_triplets,)
            relations: (num_triplets,)
        Return:

        """
        # (num_triplets, hidden_size)
        head_embedding = nodes[heads]
        rel_embedding = nodes[relations]
        # (num_triplets, 2, hidden_size)
        x = torch.stack([head_embedding, rel_embedding], dim=1)
        embedding = self._backbone(nodes, edges, x)
        return torch.sigmoid(embedding @ nodes.t())


class ConvTransR(nn.Module):
    def __init__(
        self, hidden_size: int, channels: int, kernel_size: int, dropout: float
    ):
        super().__init__()
        self._backbone = ConvTransBackbone(
            hidden_size,
            in_channels=2,
            out_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        heads: torch.Tensor,
        tails: torch.Tensor,
    ):
        """
        Arguments:
            nodes: (num_nodes, embed_size)
            edges: (num_edges, embed_size)
            heads: (num_triplets,)
            tail: (num_triplets,)
        """
        # (num_triplets, embed_size)
        head_embedding = nodes[heads]
        tail_embedding = nodes[tails]
        # (num_triplets, 2, embed_size)
        x = torch.stack([head_embedding, tail_embedding], dim=1)
        embedding = self._backbone(x)
        return torch.sigmoid(embedding @ edges.t())
