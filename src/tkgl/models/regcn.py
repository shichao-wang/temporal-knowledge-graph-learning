from typing import List, Tuple

import dgl
import tallow as tl
import torch
from dgl.udf import EdgeBatch, NodeBatch
from torch import nn

import tkgl


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
        # with graph.local_scope():
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
            self._r_linear = nn.Linear(input_size, hidden_size, bias=False)
            self._sl_linear = nn.Linear(input_size, hidden_size, bias=False)
            self._el_linear = nn.Linear(input_size, hidden_size, bias=False)
            self._rrelu = nn.RReLU()

        def forward(
            self,
            graph: dgl.DGLHeteroGraph,
        ) -> torch.Tensor:
            """
            Arguments:
                graph: dgl's Graph object
                nodes: nodes embedding (num_nodes, hidden_size)
                edges: edges embedding (num_edges, hidden_size)

            Return:
                output: (num_nodes, hidden_size)
            """

            def message_fn(edges: EdgeBatch):
                rel = edges.data["h"]
                ent = edges.src["h"]

                msg = self._r_linear(rel + ent)

                return {"msg": msg}

            def apply_fn(nodes: NodeBatch):
                h = nodes.data["h"] * nodes.data["norm"]
                h = self._rrelu(h + self_msg)
                return {"h": h}

            self_msg = self._sl_linear(graph.ndata["h"])
            isolate_nodes = torch.masked_select(
                torch.arange(0, graph.number_of_nodes()),
                (graph.in_degrees() == 0),
            )
            iso_msg = self._el_linear(graph.ndata["h"])
            self_msg[isolate_nodes] = iso_msg[isolate_nodes]
            graph.update_all(message_fn, dgl.function.sum("msg", "h"), apply_fn)
            # nodes = graph.ndata["h"] + self_msg
            # return self._rrelu(nodes)
            return graph.ndata["h"]


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
        ent_hidden: torch.Tensor,
        rel_hidden: torch.Tensor,
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
        rel_ent_embed = rel_embed
        # rel_ent_embed = torch.zeros_like(rel_embed)
        # rel_ent_embeds = torch.split_with_sizes(
        #     ent_hidden[graph.rel_node_ids], graph.rel_len
        # )
        # for rel, ent_embeds in enumerate(rel_ent_embeds):
        #     if ent_embeds.size(0) != 0:
        #         rel_ent_embed[rel] = torch.mean(ent_embeds, dim=0)

        # rel evolution
        r_rel_embed = torch.cat([rel_embed, rel_ent_embed], dim=-1)
        n_rel_embed = self._gru(r_rel_embed, rel_hidden)
        # entity evolution
        w_ent_embed = self._rgcn(graph, ent_hidden, n_rel_embed)
        u = torch.sigmoid(self._linear(ent_hidden))
        n_ent_embed = ent_hidden + u * (w_ent_embed - ent_hidden)

        return n_ent_embed, n_rel_embed


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
    ):
        super().__init__()
        self._ent_embeds = nn.Parameter(torch.zeros(num_entities, hidden_size))
        self._rel_embeds = nn.Parameter(torch.zeros(num_relations, hidden_size))
        nn.init.normal_(self._ent_embeds)
        nn.init.xavier_uniform_(self._rel_embeds)

        self._evolution = EvolutionUnit(hidden_size, hidden_size, num_layers)

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
        ent_embeds = self._ent_embeds
        rel_embeds = self._rel_embeds
        for graph in hist_graphs:
            ent_embeds, rel_embeds = self._evolution(
                graph, ent_embeds, rel_embeds, self._rel_embeds
            )
        rel_logit = self._rel_decoder(ent_embeds, rel_embeds, subj, obj)
        obj_logit = self._ent_decoder(ent_embeds, rel_embeds, subj, rel)

        return {"obj_logit": obj_logit, "rel_logit": rel_logit}
