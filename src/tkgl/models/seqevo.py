import logging
from typing import List

import dgl
import torch
from dgl.udf import EdgeBatch
from tallow.nn import forwards

from tkgl.models.evokg import RGCN

logger = logging.getLogger(__name__)


class TripletGraphConv(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph: dgl.DGLGraph):
        pass


def group_reduce_nodes(
    graph: dgl.DGLGraph,
    node_feats: torch.Tensor,
    edge_rel_ids: torch.Tensor,
    *,
    reducer: str = "mean",
):
    # num_rels = torch.Tensor.size(edge_rel_ids.unique(edge_rel_ids), 0)
    rn_mask = build_r2n_mask(graph, edge_rel_ids)

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
    edge_rel_ids: torch.Tensor,
    num_nodes: int = None,
    num_rels: int = None,
):
    if num_nodes is None:
        num_nodes = graph.num_nodes()
    if num_rels is None:
        num_rels = edge_rel_ids.max()

    rn_mask = torch.zeros(
        num_rels, num_nodes, dtype=torch.bool, device=graph.device
    )
    src, dst, eids = graph.edges("all")
    rel_ids = edge_rel_ids[eids]
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


def bi_attention(
    graph: dgl.DGLGraph, nfeats: torch.Tensor, rfeats: torch.Tensor
):
    """
    Arguments:
        nfeats: (N, H)
        rfeats: (R, H)
    """
    r2n_mask = build_r2n_mask(graph, graph.edata["rid"])
    n, _ = qkv_attention(nfeats, rfeats, rfeats, r2n_mask.t())
    r, _ = qkv_attention(rfeats, nfeats, nfeats, r2n_mask)
    return n, r


class MultiRelGraphConv(torch.nn.Module):
    def __init__(
        self, inp_sz: int, hid_sz: int, num_layers: int, dp: float = 0.0
    ):
        super().__init__()
        self._inp_layer = self.Layer(inp_sz, hid_sz)
        self._layers = torch.nn.ModuleList(
            [self.Layer(hid_sz, hid_sz) for _ in range(1, num_layers)]
        )
        self._dp = torch.nn.Dropout(dp)
        if dp != 0.0 and num_layers == 1:
            logger.warning("Invalid dropout")

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_types: torch.Tensor,
    ):
        nfeats, efeats = self._inp_layer(
            graph, node_feats, edge_feats, edge_types
        )

        for layer in self._layers:
            nfeats = self._dp(nfeats)
            efeats = self._dp(efeats)
            nfeats, efeats = layer(graph, nfeats, efeats, edge_types)
        return nfeats

    class Layer(torch.nn.Module):
        def __init__(self, inp_sz: int, hid_sz: int):
            super().__init__()
            self._linear1 = torch.nn.Linear(3 * inp_sz, hid_sz)
            self._linear2 = torch.nn.Linear(3 * inp_sz, hid_sz)
            self._linear3 = torch.nn.Linear(inp_sz, hid_sz)
            self._gru = torch.nn.GRUCell(hid_sz, hid_sz)

        def forward(
            self,
            graph: dgl.DGLGraph,
            nfeats: torch.Tensor,
            efeats: torch.Tensor,
            etype: torch.Tensor,
        ):
            def message_fn(edges: EdgeBatch):
                inp_list = [
                    rel_emb[edges.data["etype"]],
                    edges.src["h"],
                    edges.data["h"],
                ]
                msg_inp = torch.cat(inp_list, dim=-1)
                msg = self._linear1(msg_inp)
                return {"msg": msg}

            def update_edges(edges: EdgeBatch):
                # (num_edges, hid)
                inp_list = [
                    rel_emb[edges.data["etype"]],
                    edges.src["h"],
                    edges.dst["h"],
                ]
                edges_inp = torch.cat(inp_list, dim=-1)
                edges.data["msg"] = self._linear2(edges_inp)

            rel_emb = group_reduce_nodes(graph, nfeats, etype)
            with graph.local_scope():
                graph.ndata["h"] = nfeats
                graph.edata["h"] = efeats
                graph.edata["etype"] = etype

                # node msg
                graph.update_all(message_fn, dgl.function.mean("msg", "msg"))
                # edge msg
                graph.apply_edges(update_edges)

                node_msg = graph.ndata["msg"]
                edge_msg = graph.edata["msg"]

            node_feats = torch.rrelu(node_msg + self._linear3(nfeats))
            edge_feats = self._gru(edge_msg, efeats)

            return node_feats, edge_feats


class SeqEvo(torch.nn.Module):
    def __init__(
        self,
        num_ents: int,
        num_rels: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self._mrgcn = MultiRelGraphConv(
            hidden_size, hidden_size, num_layers, dropout
        )
        self._gru1 = torch.nn.GRUCell(
            2 * hidden_size, hidden_size, batch_first=True
        )
        self._gru2 = torch.nn.GRUCell(
            2 * hidden_size, hidden_size, batch_first=True
        )

        self._linear1 = torch.nn.Linear(hidden_size, hidden_size)

        self.ent_emb = torch.nn.Parameter(torch.zeros(num_ents, hidden_size))
        self.rel_emb = torch.nn.Parameter(torch.zeros(num_rels, hidden_size))
        torch.nn.init.xavier_normal_(self.ent_emb)
        torch.nn.init.xavier_uniform_(self.rel_emb)

    def forward(
        self,
        hist_graphs: List[dgl.DGLGraph],
        subj: torch.Tensor,
        rel: torch.Tensor,
        obj: torch.Tensor,
    ):
        bg = dgl.batch(hist_graphs)
        total_nfeats = self._rgcn(bg, self.ent_emb[bg.ndata["eid"]])
        hist_nfeat_list = torch.split_with_sizes(
            total_nfeats, bg.batch_num_nodes().tolist()
        )
        hist_nfeats = torch.stack(hist_nfeat_list, dim=1)
        # (num_ents, hist_len, hidden_size)
        hist_nhiddens, _ = self._gru(hist_nfeats, self.ent_emb.unsqueeze(1))
        # hist_gfeats, _ = self._gru(hist_gfeats)
        # transformer_hist_nfeats = forwards.transformer_encoder_forward(
        #     self._transformer_encoder, hist_nfeats
        # )
        # transformer_nfeats = self._linear1(transformer_hist_nfeats[:, -1, :])

    def recursive_reasoning(self, hist_graphs: List[dgl.DGLGraph]):
        ent_emb = self.ent_emb
        rel_emb = self.rel_emb
        for graph in hist_graphs:
            nfeats = self._mrgcn(
                graph,
                ent_emb[graph.ndata["eid"]],
                rel_emb[graph.edata["rid"]],
                graph.edata["rid"],
            )
            n_emb = nfeats[torch.argsort(graph.ndata["eid"])]
            r_emb = rel_emb
            nr_feats, rn_feats = bi_attention(graph, n_emb, r_emb)
            ent_emb = self._gru1(torch.cat([nr_feats, n_emb], dim=-1), ent_emb)
            rel_emb = self._gru2(torch.cat([rn_feats, r_emb], dim=-1), rel_emb)

        return ent_emb, rel_emb

    def highway_reasoning(self, hist_graphs: List[dgl.DGLGraph]):
        bg = dgl.batch(hist_graphs)
        total_nfeats = self._rgcn(bg, self.ent_emb[bg.ndata["eid"]])
        hist_nfeat_list = torch.split_with_sizes(
            total_nfeats, bg.batch_num_nodes().tolist()
        )
        hist_nfeats = torch.stack(hist_nfeat_list, dim=1)
        # (num_ents, hist_len, hidden_size)
        hist_nhiddens, _ = self._gru(hist_nfeats, self.ent_emb.unsqueeze(1))
        pass
