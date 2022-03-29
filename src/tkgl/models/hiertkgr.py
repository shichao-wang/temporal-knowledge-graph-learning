import logging
from typing import Callable, List, Tuple

import dgl
import torch
from dgl.udf import EdgeBatch
from numpy import tri
from tallow.nn import forwards, positional_encoding
from torch.nn import functional as f

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
    """
    Returns:
        (T, 3, H)
    """

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
        dropout: float,
    ):
        super().__init__()
        layer = MultiRelGraphLayer
        _layers = [layer(input_sizse, hidden_size, dropout)]
        for _ in range(1, num_layers):
            _layers.append(layer(hidden_size, hidden_size, dropout))
        self._layers = torch.nn.ModuleList(_layers)
        self._linear = torch.nn.Linear(
            hidden_size * (1 + num_layers), hidden_size
        )

    __call__: Callable[
        [
            "MultiRelGraphConv",
            dgl.DGLGraph,
            torch.Tensor,
            torch.Tensor,
            torch.LongTensor,
        ],
        Tuple[torch.Tensor, torch.Tensor],
    ]

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ):
        msg_box = [node_feats]
        for layer in self._layers:
            neighbor_msg = layer(graph, msg_box[-1], edge_feats)
            msg_box.append(neighbor_msg)

        multihop_neighbor = torch.cat(msg_box, dim=-1)
        return self._linear(multihop_neighbor)


class MultiRelGraphLayer(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self._linear1 = torch.nn.Linear(2 * input_size, hidden_size)
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
            msg_inp = torch.cat([edges.src["h"], edges.data["h"]], dim=-1)
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
        node_feats = node_msg

        self_msg = self._linear2(node_feats)
        node_feats = node_feats + self_msg

        return self._activation(node_feats)


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
        self.mrgcn = MultiRelGraphConv(
            hidden_size, hidden_size, num_layers, dropout
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(hidden_size, 4, dropout=dropout),
            num_layers=1,
        )
        self.slinear = torch.nn.Linear(hidden_size, hidden_size)
        self.ssa = StructuredSA(hidden_size, 4, reduction="concat")
        self.qlinear = torch.nn.Linear(2 * hidden_size, hidden_size)
        self.convtranse = ConvTransE(hidden_size, 2, 50, 3, dropout)

    def forward(
        self,
        hist_graphs: List[dgl.DGLGraph],
        subj: torch.Tensor,
        rel: torch.Tensor,
    ):
        ent_emb = f.normalize(self.ent_emb)
        rel_emb = f.normalize(self.rel_emb)
        bg = dgl.batch(hist_graphs)
        total_neighbor_emb = self.mrgcn(
            bg, ent_emb[bg.ndata["eid"]], rel_emb[bg.edata["rid"]]
        )
        # sf means seqlen first
        neighbor_emb_lf = torch.stack(
            torch.split_with_sizes(
                total_neighbor_emb, bg.batch_num_nodes().tolist()
            ),
            dim=0,
        )
        seq_len = len(hist_graphs)
        mask = torch.triu(
            neighbor_emb_lf.new_ones((seq_len, seq_len), dtype=torch.bool)
        )
        hist_ent_emb = self.transformer_encoder(
            hist_ent_emb + positional_encoding(hist_ent_emb),
            mask=~mask,
        )
        hist_ent_emb = torch.transpose(hist_ent_emb, 0, 1)

        evolve_ent_emb, _ = self.ssa(hist_ent_emb)
        q = torch.stack([evolve_ent_emb[subj], self.rel_emb[rel]], dim=1)
        obj_logit = self.convtranse(q) @ evolve_ent_emb.t()
        return {"obj_logit": obj_logit}


class HierTkgr(TkgrModel):
    def __init__(
        self,
        num_ents: int,
        num_rels: int,
        hidden_size: int,
        dropout: float,
        rgcn_num_layers: int,
        rgcn_self_loop: bool,
    ) -> None:
        super().__init__(num_ents, num_rels, hidden_size)

        self.rgcn = OmegaRelGraphConv(
            hidden_size, hidden_size, rgcn_num_layers, rgcn_self_loop, dropout
        )
        self.glinear = torch.nn.Linear(hidden_size, hidden_size)
        self.rel_rnn = torch.nn.GRUCell(2 * hidden_size, hidden_size)
        self.qlinear = torch.nn.Linear(2 * hidden_size, hidden_size)
        self.trip_linear = torch.nn.Linear(3 * hidden_size, hidden_size)
        self.trip_rnn = torch.nn.GRU(
            hidden_size, hidden_size, batch_first=False
        )
        convtranse_channels = 50
        convtranse_kernel_size = 3
        self.convtranse = ConvTransE(
            hidden_size,
            3,
            convtranse_channels,
            convtranse_kernel_size,
            dropout,
        )

    def evolve(self, hist_graphs: List[dgl.DGLGraph]):
        ent_emb = f.normalize(self.ent_emb)
        rel_emb = self.rel_emb

        ent_emb_list = []
        rel_emb_list = []
        for graph in hist_graphs:
            node_feats = ent_emb[graph.ndata["eid"]]
            # relation evolution
            rel_ent_emb = group_reduce_nodes(
                graph, ent_emb, graph.edata["rid"], self.num_ents, self.num_rels
            )
            gru_input = torch.cat([self.rel_emb, rel_ent_emb], dim=-1)
            rel_emb = self.rel_rnn(gru_input, rel_emb)
            rel_emb = f.normalize(rel_emb)
            # entity evolution
            edge_feats = rel_emb[graph.edata["rid"]]
            neigh_feats = self.rgcn(graph, node_feats, edge_feats)
            neigh_feats = f.normalize(neigh_feats)
            cur_ent_emb = neigh_feats[torch.argsort(graph.ndata["eid"])]
            u = torch.sigmoid(self.glinear(cur_ent_emb))
            ent_emb = u * cur_ent_emb + (1 - u) * ent_emb
            ent_emb = f.normalize(ent_emb)

            ent_emb_list.append(ent_emb)
            rel_emb_list.append(rel_emb)

        hist_ent_emb = torch.stack(ent_emb_list)
        hist_rel_emb = torch.stack(rel_emb_list)
        return hist_ent_emb, hist_rel_emb

    def forward(
        self,
        hist_graphs: List[dgl.DGLGraph],
        subj: torch.Tensor,
        rel: torch.Tensor,
        # obj: torch.Tensor,
    ):
        hist_ent_emb, hist_rel_emb = self.evolve(hist_graphs)
        ent_emb = hist_ent_emb[-1]
        rel_emb = hist_rel_emb[-1]
        queries = self.qlinear(torch.cat([ent_emb[subj], rel_emb[rel]], dim=1))
        triplets_list = [
            graph_to_triplets(hist_graphs[i], hist_ent_emb[i], hist_rel_emb[i])
            for i in range(len(hist_graphs))
        ]
        hist_triplets = torch.cat(triplets_list, dim=0)
        num_triplets = hist_triplets.new_tensor(
            [trip.size(0) for trip in triplets_list], dtype=torch.long
        )
        trip_repr = self.trip_linear(
            hist_triplets.view(hist_triplets.size(0), -1)
        )
        trip_scores = trip_repr @ queries.t()
        # (T, Q)
        trip_weights = dgl.ops.segment_softmax(num_triplets, trip_scores)

        trip_emb_list = []
        i = 0
        for length in num_triplets:
            trip_emb = (
                trip_weights[i : i + length].t() @ trip_repr[i : i + length]
            )
            trip_emb_list.append(trip_emb)
            i = i + length
        hist_trip_emb = torch.stack(trip_emb_list)
        hist_trip_hid, _ = self.trip_rnn(hist_trip_emb)

        pred_inp = torch.stack(
            [ent_emb[subj], rel_emb[rel], hist_trip_hid[-1]], dim=1
        )
        obj_logit = self.convtranse(pred_inp) @ ent_emb.t()
        return {"obj_logit": obj_logit}


class StructuredSA(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        num_heads: int,
        internal_size: int = None,
        reduction: str = "none",
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
        self.reduction = reduction
        if self.reduction == "concat":
            self.rlinear = torch.nn.Linear(num_heads * input_size, input_size)

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
        heads_align = (weightsT @ weights) - torch.eye(
            self.num_heads, device=weights.device, dtype=torch.float
        )
        penalty = torch.norm(heads_align)
        hidden = weightsT @ inputs
        if self.reduction == "concat":
            hidden = self.rlinear(hidden.flatten(-2))
        elif self.reduction == "mean":
            hidden = torch.mean(hidden, dim=1)
        return hidden, penalty
