import logging
from typing import Callable, List, Tuple

import dgl
import torch
from dgl.udf import EdgeBatch
from tallow.nn import forwards

from tkgl.convtranse import ConvTransE
from tkgl.models.regcn import OmegaRelGraphConv
from tkgl.models.tkgr_model import TkgrModel
from tkgl.modules.compgcn import CompGCN
from tkgl.modules.rgat import RelGat

logger = logging.getLogger(__name__)


def group_reduce_nodes(
    graph: dgl.DGLGraph,
    node_feats: torch.Tensor,
    edge_types: torch.Tensor,
    num_nodes: int = None,
    num_rels: int = None,
    *,
    reducer: str = "mean",
):
    rn_mask = build_r2n_mask(graph, edge_types, num_nodes, num_rels)

    nids = torch.nonzero(rn_mask)[:, 1]
    rel_emb = dgl.ops.segment_reduce(
        rn_mask.sum(dim=1), node_feats[nids], reducer
    )
    return torch.nan_to_num(rel_emb, 0)


def graph_to_triplets(
    graph: dgl.DGLGraph, ent_embeds: torch.Tensor, rel_embeds: torch.Tensor
) -> torch.Tensor:

    src, dst, eid = graph.edges("all")
    subj = graph.ndata["eid"][src]
    rel = graph.edata["rid"][eid]
    obj = graph.ndata["eid"][dst]
    embed_list = [ent_embeds[subj], rel_embeds[rel], ent_embeds[obj]]
    return torch.stack(embed_list, dim=1)


def build_r2n_mask(
    graph: dgl.DGLGraph,
    edge_types: torch.Tensor,
    num_nodes: int = None,
    num_rels: int = None,
):
    if num_nodes is None:
        num_nodes = graph.num_nodes()
    if num_rels is None:
        num_rels = edge_types.max()

    rn_mask = torch.zeros(
        num_rels, num_nodes, dtype=torch.bool, device=graph.device
    )
    src, dst, eids = graph.edges("all")
    rel_ids = edge_types[eids]
    rn_mask[rel_ids, src] = True
    rn_mask[rel_ids, dst] = True
    return rn_mask


def qkv_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qk_mask: torch.Tensor = None,
):
    """
    Arguments:
        q: (Q, H)
        k: (KV, H)
        v: (KV, H)
        qk_mask: (Q, KV)
    """
    d = q.size(-1)
    s = (q @ k.t()) / torch.sqrt(d)  # (Q, KV)
    if qk_mask is not None:
        s.masked_fill_(~qk_mask, float("-inf"))
    a = s.softmax(s, dim=-1)
    return a @ v, a


class MultiRelGraphConv(torch.nn.Module):
    def __init__(
        self,
        input_sizse: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
    ):
        super().__init__()
        layer = MultiRelGraphLayer
        self._inp_layer = layer(input_sizse, hidden_size, activation, dropout)
        self._layers = torch.nn.ModuleList(
            [
                layer(hidden_size, hidden_size, activation, dropout)
                for _ in range(1, num_layers)
            ]
        )

    __call__: Callable[
        [
            "MultiRelGraphConv",
            dgl.DGLGraph,
            torch.Tensor,
            torch.Tensor,
            torch.LongTensor,
        ],
        torch.Tensor,
    ]

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_types: torch.Tensor,
    ):
        nfeats = self._inp_layer(graph, node_feats, edge_feats, edge_types)
        for layer in self._layers:
            nfeats = layer(graph, nfeats, edge_feats, edge_types)
        return nfeats


class MultiRelGraphLayer(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
        dropout: float = 0.0,
    ):
        super().__init__()
        self._linear1 = torch.nn.Linear(3 * input_size, hidden_size)
        self._linear2 = torch.nn.Linear(3 * input_size, hidden_size)
        self._gru = torch.nn.GRUCell(hidden_size, hidden_size)
        self._activation = activation
        self._dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ):
        def message_fn(edges: EdgeBatch):
            inp_list = [
                edges.src["h"],
                edges.data["h"],
            ]
            msg_inp = torch.cat(inp_list, dim=-1)
            msg = self._linear1(msg_inp)
            return {"msg": msg}

        node_feats = self._dropout(node_feats)
        edge_feats = self._dropout(edge_feats)
        with graph.local_scope():
            graph.ndata["h"] = node_feats
            graph.edata["h"] = edge_feats
            # node msg
            graph.update_all(message_fn, dgl.function.mean("msg", "msg"))

            node_msg = graph.ndata["msg"]

        return self._activation(node_msg + node_feats)


class HighwayHierTkgr(TkgrModel):
    def __init__(
        self,
        num_ents: int,
        num_rels: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__(num_ents, num_rels, hidden_size)
        self.mrgcn = RelGat(hidden_size, hidden_size, num_layers, dropout)
        self.gru1 = torch.nn.GRU(hidden_size, hidden_size, batch_first=False)
        self.gru2 = torch.nn.GRU(hidden_size, hidden_size, batch_first=False)
        self.gru3 = torch.nn.GRU(hidden_size, hidden_size, batch_first=False)
        self.linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self.convtranse = ConvTransE(hidden_size, 3, 50, 3, dropout)

    def forward(
        self,
        hist_graphs: List[dgl.DGLGraph],
        subj: torch.Tensor,
        rel: torch.Tensor,
    ):
        bg = dgl.batch(hist_graphs)
        total_nfeats: torch.Tensor = self.mrgcn(
            bg, self.ent_emb[bg.ndata["eid"]], self.rel_emb[bg.edata["rid"]]
        )
        hist_num_nodes = bg.batch_num_nodes().tolist()
        hist_nfeats = torch.stack(
            torch.split_with_sizes(total_nfeats, hist_num_nodes),
            dim=0,
        )
        hist_ent_embs, ent_emb = self.gru1(
            hist_nfeats, self.ent_emb.unsqueeze(0)
        )
        queries = ent_emb[subj] + self.rel_emb[rel]

        total_trips = self.linear1(
            graph_to_triplets(bg, self.ent_emb, self.rel_emb).flatten(-2)
        )
        # (Q, T)
        trip_scores = queries @ total_trips.t()
        trip_weights = dgl.ops.segment_softmax(
            bg.batch_num_edges(), trip_scores.t()
        ).t()
        # (Q, T, H)
        x = torch.unsqueeze(trip_weights, dim=-1) * total_trips
        hist_trip_hidden = dgl.ops.segment_reduce(
            bg.batch_num_nodes(), torch.transpose(x, 0, 1)
        )
        _, trip_hidden = self.gru2(hist_trip_hidden)

        # （L, Q, N)
        hist_glob_scores = queries @ hist_ent_embs.transpose(-1, -2)
        hist_glob_weights = torch.softmax(hist_glob_scores, dim=-1)
        hist_glob_hidden = hist_glob_weights @ hist_ent_embs
        _, glob_hidden = self.gru3(hist_glob_hidden)

        pred_inp = torch.stack(
            [ent_emb[subj], self.rel_emb[rel], trip_hidden, glob_hidden], dim=1
        )
        obj_logit = self.convtranse(pred_inp, ent_emb)
        return {"obj_logit": obj_logit}


class HierTKGR(TkgrModel):
    def __init__(
        self,
        num_ents: int,
        num_rels: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__(num_ents, num_rels, hidden_size)

        self._mrgcn = RelGat(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self._linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self._linear2 = torch.nn.Linear(3 * hidden_size, hidden_size)
        self._linear3 = torch.nn.Linear(hidden_size, hidden_size)
        self._linear4 = torch.nn.Linear(hidden_size, hidden_size)
        self._rnn1 = torch.nn.GRU(hidden_size, hidden_size, batch_first=True)
        self._rnn2 = torch.nn.GRU(hidden_size, hidden_size, batch_first=True)

        self.convtranse = ConvTransE(hidden_size, 4, 50, 3, dropout)

    def forward(
        self,
        hist_graphs: List[dgl.DGLGraph],
        subj: torch.Tensor,
        rel: torch.Tensor,
        # obj: torch.Tensor,
    ):
        ent_emb = self.ent_emb
        rel_emb = self.rel_emb
        hist_trip_list = []
        hist_global_list = []
        for graph in hist_graphs:
            edge_feats = rel_emb[graph.edata["rid"]]
            node_feats = self._mrgcn(graph, ent_emb, edge_feats)
            u = torch.sigmoid(self._linear1(ent_emb))
            ent_emb = ent_emb + u * (node_feats - ent_emb)

            # queries = self._linear3(
            #     torch.cat([ent_emb[subj], rel_emb[rel]], dim=-1)
            # )
            queries = ent_emb[subj] + rel_emb[rel]
            # semantic
            triples = self._linear2(
                graph_to_triplets(graph, ent_emb, rel_emb).flatten(-2)
            )

            trip_weights = torch.softmax(
                self._linear3(queries) @ triples.t(), dim=-1
            )
            trip_hidden = trip_weights @ triples

            glob_weights = torch.softmax(
                self._linear4(queries) @ ent_emb.t(), dim=-1
            )
            glob_hidden = glob_weights @ ent_emb

            hist_trip_list.append(trip_hidden)
            hist_global_list.append(glob_hidden)

        hist_trip = torch.stack(hist_trip_list, dim=1)
        _, triplet_hidden = forwards.rnn_forward(self._rnn1, hist_trip)

        hist_global = torch.stack(hist_global_list, dim=1)
        _, global_hidden = forwards.rnn_forward(self._rnn2, hist_global)

        pred_inp = torch.stack(
            [
                ent_emb[subj],
                rel_emb[rel],
                triplet_hidden,
                global_hidden,
            ],
            dim=1,
        )
        obj_logit = self.convtranse(pred_inp, ent_emb)

        return {"obj_logit": obj_logit}


class StructuredSA(torch.nn.Module):
    def __init__(
        self, input_size: int, num_heads: int, internal_size: int = None
    ):
        super().__init__()
        if internal_size is None:
            internal_size = input_size
        self.num_heads = num_heads
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(input_size, internal_size),
            torch.nn.Tanh(),
            torch.nn.Linear(internal_size, num_heads),
        )

    __call__: Callable[
        ["StructuredSA", torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
    ]

    def forward(self, inputs: torch.Tensor):
        """
        Arguments:
            inputs: (*, seq_len, input_size)
        Return:
            (*, num_heads, input_size)
            ()
        """
        scores = self.linear(inputs)
        weights = torch.softmax(scores, dim=-2)
        weightsT = weights.transpose(-1, -2)
        heads_align = (weightsT @ weights) - weights.new_tensor(
            torch.eye(self.num_heads)
        )
        penalty = torch.norm(heads_align)
        hidden = weightsT @ inputs
        return hidden, penalty
