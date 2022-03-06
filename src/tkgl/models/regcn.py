from typing import List, Tuple

import dgl
import torch
from dgl.udf import EdgeBatch
from torch import nn
from torch.nn import functional as tnf


class OmegaRelGraphConv(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float
    ):
        super().__init__()

        self._layers = nn.ModuleList(
            [self.Layer(input_size, hidden_size, dropout)]
        )
        for _ in range(1, num_layers):
            self._layers.append(self.Layer(hidden_size, hidden_size, dropout))

    def forward(
        self,
        graph: dgl.DGLHeteroGraph,
        features: Tuple[torch.Tensor],
    ) -> torch.Tensor:
        """
        Arguments:
            graph: dgl.DGLGraph
            ent_embeds: (num_nodes, input_size)
            rel_embeds: (num_edges, input_size)
        """
        for layer in self._layers:
            features = layer(graph, features)

        return features

    class Layer(nn.Module):
        """
        Notice:
        This implementation of RGCN Layer is not equivalent to the one decribed in the paper.
        In the paper, there is another self-evolve weight matrix(nn.Linear) for those entities do not exist in current graph.
        We migrate it with `self._loop_linear` here for simplicity.
        """

        def __init__(self, input_size: int, hidden_size: int, dropout: float):
            super().__init__()
            self._linear1 = nn.Linear(input_size, hidden_size, bias=False)
            self._linear2 = nn.Linear(input_size, hidden_size, bias=False)
            self._linear3 = nn.Linear(input_size, hidden_size, bias=False)
            self._dp = nn.Dropout(dropout)

        def forward(
            self,
            graph: dgl.DGLHeteroGraph,
            features: Tuple[torch.Tensor, torch.Tensor],
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
                graph.ndata["h"], graph.edata["h"] = features

                self_msg = self._linear2(graph.ndata["h"])
                isolate_nids = torch.masked_select(
                    torch.arange(0, graph.number_of_nodes()),
                    (graph.in_degrees() == 0),
                )
                iso_msg = self._linear3(graph.ndata["h"])
                self_msg[isolate_nids] = iso_msg[isolate_nids]

                graph.update_all(message_fn, dgl.function.mean("msg", "h"))

                node_feats: torch.Tensor = graph.ndata["h"]
                node_feats = torch.rrelu(node_feats + self_msg)
                node_feats = self._dp(node_feats)
                return node_feats, graph.edata["h"]


class ConvTransE(torch.nn.Module):
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
            padding=(kernel_size - 1) // 2,
        )
        self._bn1 = nn.BatchNorm1d(out_channels)
        self._dp1 = nn.Dropout(dropout)
        self._linear = nn.Linear(hidden_size * out_channels, hidden_size)
        self._bn2 = nn.BatchNorm1d(hidden_size)
        self._dp2 = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor):
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
        return self._bn2(self._dp2(hidden)).relu()
        # (num_triplets, embed_size)


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

        self._rgcn = OmegaRelGraphConv(
            hidden_size, hidden_size, num_layers, dropout
        )
        self._linear = nn.Linear(hidden_size, hidden_size)
        self._gru = nn.GRUCell(2 * hidden_size, hidden_size)

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
        for graph in hist_graphs:
            # rel evolution
            # rel_ent_embeds = self.rel_embeds
            rel_ent_embeds = self._agg_rel_nodes(graph, ent_embeds)
            gru_input = torch.cat([rel_ent_embeds, self.rel_embeds], dim=-1)
            n_rel_embeds = self._gru(gru_input, rel_embeds)
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
