import logging
from typing import Callable, List, Tuple

import dgl
import torch
from dgl.udf import EdgeBatch
from numpy import tri
from tallow.nn import forwards, positional_encoding
from torch.nn import functional as f

from tkgl.models.regcn import OmegaRelGraphConv
from tkgl.models.tkgr_model import TkgrModel
from tkgl.modules.compgcn import CompGCN
from tkgl.modules.convtranse import ConvTransE
from tkgl.modules.mrgcn import MultiRelGraphConv
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
    in_degree, out_degree = build_r2n_degree(
        graph, edge_types, num_rels, num_nodes
    )
    rn_mask = torch.bitwise_or(in_degree.bool(), out_degree.bool())

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


def build_r2n_degree(
    graph: dgl.DGLGraph,
    edge_types: torch.Tensor,
    num_rels: int,
    num_nodes: int = None,
):
    if num_nodes is None:
        num_nodes = graph.num_nodes()

    in_degrees = torch.zeros(
        num_rels, num_nodes, dtype=torch.long, device=graph.device
    )
    out_degrees = torch.zeros(
        num_rels, num_nodes, dtype=torch.long, device=graph.device
    )
    src, dst, eids = graph.edges("all")
    rel_ids = edge_types[eids]
    in_degrees[rel_ids, src] += 1
    out_degrees[rel_ids, dst] += 1
    return in_degrees, out_degrees


class HighwayTkgr(TkgrModel):
    def __init__(
        self,
        num_ents: int,
        num_rels: int,
        hidden_size: int,
        rgcn_num_layers: int,
        rgcn_num_heads: bool,
        dropout: float,
    ):
        super().__init__(num_ents, num_rels, hidden_size)
        self.mrgcn = MultiRelGraphConv(
            hidden_size, hidden_size, rgcn_num_layers, rgcn_num_heads, dropout
        )
        self.relrnn = torch.nn.GRUCell(2 * hidden_size, hidden_size)
        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(hidden_size, 4, dropout=dropout),
            num_layers=2,
        )
        self.tuplinear = torch.nn.Linear(2 * hidden_size, hidden_size)
        self.convtranse = ConvTransE(hidden_size, 2, 50, 3, dropout)

    def forward(
        self,
        hist_graphs: List[dgl.DGLGraph],
        subj: torch.Tensor,
        rel: torch.Tensor,
    ):
        ent_emb = f.normalize(self.ent_emb)

        hist_ent_emb_list = []
        hist_rel_emb_list = []
        for graph in hist_graphs:
            # Update for connected nodes
            node_feats = self.mrgcn(
                graph,
                ent_emb[graph.ndata["eid"]],
                self.rel_emb[graph.edata["rid"]],
            )
            g_ent_emb = node_feats[torch.argsort(graph.ndata["eid"])]
            hist_ent_emb_list.append(g_ent_emb)

            in_mask, out_mask = build_r2n_degree(
                graph, graph.edata["rid"], self.num_rels
            )
            in_hid = (in_mask @ ent_emb) / torch.sum(
                in_mask, dim=1, keepdim=True
            )
            out_hid = (out_mask @ ent_emb) / torch.sum(
                out_mask, dim=1, keepdim=True
            )
            e = torch.cat([in_hid, out_hid], dim=-1)
            g_rel_emb = self.relrnn(e, self.rel_emb)
            hist_rel_emb_list.append(g_rel_emb)

        hist_ent_emb = torch.stack(hist_ent_emb_list)
        hist_rel_emb = torch.stack(hist_rel_emb_list)

        hist_len = len(hist_graphs)
        mask = torch.triu(
            hist_ent_emb.new_ones((hist_len, hist_len), dtype=torch.bool)
        )
        hist_ent_emb = self.transformer_encoder(hist_ent_emb, mask=~mask)
        hist_obj_logit_list = []
        for i in range(hist_len):
            ent_emb = hist_ent_emb[i]
            rel_emb = hist_rel_emb[i]
            obj_inp = torch.stack([ent_emb[subj], rel_emb[rel]], dim=1)
            obj_logit = self.convtranse(obj_inp) @ self.ent_emb.t()
            hist_obj_logit_list.append(obj_logit)

        hist_obj_logit = torch.stack(hist_obj_logit_list)
        obj_logit = hist_obj_logit.softmax(dim=-1).sum(dim=0)

        return {
            "hist_obj_logit": hist_obj_logit,
            "hist_ent_emb": hist_ent_emb,
            "obj_logit": obj_logit,  # used for prediction
        }


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
        for graph in hist_graphs:
            node_feats = ent_emb[graph.ndata["eid"]]
            # entity evolution
            edge_feats = rel_emb[graph.edata["rid"]]
            neigh_feats = self.rgcn(graph, node_feats, edge_feats)
            neigh_feats = f.normalize(neigh_feats)
            cur_ent_emb = neigh_feats[torch.argsort(graph.ndata["eid"])]
            u = torch.sigmoid(self.glinear(cur_ent_emb))
            ent_emb = f.normalize(u * cur_ent_emb + (1 - u) * ent_emb)

            ent_emb_list.append(ent_emb)

        hist_ent_emb = torch.stack(ent_emb_list)
        return hist_ent_emb

    def forward(
        self,
        hist_graphs: List[dgl.DGLGraph],
        subj: torch.Tensor,
        rel: torch.Tensor,
        # obj: torch.Tensor,
    ):
        hist_ent_emb = self.evolve(hist_graphs)

        trip_hid_list = []
        glob_hid_list = []
        for i, graph in enumerate(hist_graphs):
            ent_emb = hist_ent_emb[i]
            queries = torch.cat([ent_emb[subj], self.rel_emb[rel]], dim=1)

            # trip
            triplets = graph_to_triplets(graph, ent_emb, self.rel_emb)
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
        pred_inp = torch.stack([ent_emb[subj], self.rel_emb[rel]], dim=1)
        kg_logit = self.convtranse(pred_inp) @ ent_emb.t()
        trip_logit = hist_trip_hid[-1].relu() @ ent_emb.t()
        glob_logit = hist_glob_hid[-1].relu() @ ent_emb.t()
        return {"obj_logit": kg_logit + glob_logit + trip_logit}


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
