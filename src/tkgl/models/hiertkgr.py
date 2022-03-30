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
        neigh_msg_list = []
        neigh_feats = node_feats
        for layer in self.layers:
            neigh_feats = layer(graph, neigh_feats, edge_feats)
            neigh_msg_list.append(neigh_feats)

        multihop_neighbor = torch.cat(neigh_msg_list, dim=-1)
        return self.olinear(multihop_neighbor)


class MultiRelGraphLayer(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        self_loop: bool,
        dropout: float = 0.0,
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

        node_feats = self._dropout(node_feats)
        edge_feats = self._dropout(edge_feats)
        with graph.local_scope():
            graph.ndata["h"] = node_feats
            graph.edata["h"] = edge_feats
            # node msg
            graph.update_all(message_fn, dgl.function.mean("msg", "msg"))

            neigh_msg = graph.ndata["msg"]

        if self._self_loop:
            self_msg = self.looplinear(node_feats)
            neigh_msg = neigh_msg + self_msg
        return self._activation(node_feats)


class HighwayHierTkgr(TkgrModel):
    def __init__(
        self,
        num_ents: int,
        num_rels: int,
        hidden_size: int,
        rgcn_num_layers: int,
        rgcn_self_loop: bool,
        dropout: float,
    ):
        super().__init__(num_ents, num_rels, hidden_size)
        torch.nn.init.xavier_uniform_(self.ent_emb)
        torch.nn.init.xavier_uniform_(self.rel_emb)
        self.mrgcn = MultiRelGraphConv(
            hidden_size, hidden_size, rgcn_num_layers, rgcn_self_loop, dropout
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
        # lf means seqlen first
        hist_ent_neigh_emb_lf = torch.stack(
            torch.split_with_sizes(
                total_neighbor_emb, bg.batch_num_nodes().tolist()
            ),
            dim=0,
        )
        seq_len = len(hist_graphs)
        mask = torch.triu(
            hist_ent_neigh_emb_lf.new_ones((seq_len, seq_len), dtype=torch.bool)
        )
        hist_ent_neigh_emb_lf = f.normalize(
            hist_ent_neigh_emb_lf + ent_emb.unsqueeze(dim=0)
        )
        hist_ent_emb = self.transformer_encoder(
            hist_ent_neigh_emb_lf + positional_encoding(hist_ent_neigh_emb_lf),
            mask=~mask,
        )
        evolve_ent_emb = torch.mean(hist_ent_emb, dim=0)
        q = torch.stack([evolve_ent_emb[subj], self.rel_emb[rel]], dim=1)
        obj_logit = self.convtranse(q) @ self.ent_emb.t()
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
        torch.nn.init.xavier_uniform_(self.ent_emb)
        torch.nn.init.xavier_uniform_(self.rel_emb)

        self.rgcn = MultiRelGraphConv(
            hidden_size, hidden_size, rgcn_num_layers, rgcn_self_loop, dropout
        )
        self.glinear = torch.nn.Linear(hidden_size, hidden_size)
        self.rel_rnn = torch.nn.GRUCell(2 * hidden_size, hidden_size)
        self.trip_qlinear = torch.nn.Linear(2 * hidden_size, hidden_size)
        self.trip_linear = torch.nn.Linear(3 * hidden_size, hidden_size)
        self.trip_rnn = torch.nn.GRU(
            hidden_size, hidden_size, batch_first=False
        )
        self.glob_rnn = torch.nn.GRU(
            hidden_size, hidden_size, batch_first=False
        )
        convtranse_channels = 50
        convtranse_kernel_size = 3
        self.convtranse = ConvTransE(
            hidden_size,
            2,
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

        trip_hid_list = []
        glob_hid_list = []
        for i, graph in enumerate(hist_graphs):
            ent_emb = hist_ent_emb[i]
            rel_emb = hist_rel_emb[i]
            queries = torch.cat([ent_emb[subj], rel_emb[rel]], dim=1)

            # trip
            triplets = graph_to_triplets(graph, ent_emb, rel_emb)
            trip_emb = self.trip_linear(triplets.view(triplets.size(0), -1))
            trip_weights = torch.softmax(
                self.trip_qlinear(queries) @ trip_emb.t(), dim=-1
            )
            trip_hid = trip_weights @ trip_emb
            trip_hid_list.append(trip_hid)

            # global
            glob_emb = torch.mean(ent_emb, dim=0)
            glob_hid_list.append(
                glob_emb.unsqueeze(0).expand(queries.size(0), -1)
            )

        hist_trip_hid = torch.stack(trip_hid_list)
        hist_trip_hid, _ = self.trip_rnn(hist_trip_hid)

        hist_glob_hid = torch.stack(glob_hid_list)
        hist_glob_hid, _ = self.glob_rnn(hist_glob_hid)

        ent_emb = hist_ent_emb[-1]
        rel_emb = hist_rel_emb[-1]
        pred_inp = torch.stack([ent_emb[subj], rel_emb[rel]], dim=1)
        kg_logit = self.convtranse(pred_inp) @ ent_emb.t()
        # trip_logit = hist_trip_hid[-1].relu() @ ent_emb.t()
        # glob_logit = hist_glob_hid[-1].relu() @ ent_emb.t()
        return {"obj_logit": kg_logit}  # + glob_logit}


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
