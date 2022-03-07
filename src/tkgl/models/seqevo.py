import logging
from typing import List

import dgl
import torch
from dgl.udf import EdgeBatch

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
    num_rels = edge_rel_ids.max()
    # (num_rels, num_nodes)
    rn_mask = node_feats.new_zeros(
        num_rels, node_feats.size(0), dtype=torch.bool
    )
    src, dst, eids = graph.edges("all")
    rel_ids = edge_rel_ids[eids]
    rn_mask[rel_ids, src] = True
    rn_mask[rel_ids, dst] = True

    nids = torch.nonzero(rn_mask)[:, 1]
    rel_emb = dgl.ops.segment_reduce(
        rn_mask.sum(dim=1), node_feats[nids], reducer
    )
    return torch.nan_to_num(rel_emb, 0)


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
        return nfeats, efeats

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
                edges_msg = self._linear2(edges_inp)
                edges.data["msg"] = self._gru(edges_msg, edges.data["h"])

            rel_emb = group_reduce_nodes(graph, nfeats, etype)
            with graph.local_scope():
                graph.ndata["h"] = nfeats
                graph.edata["h"] = efeats
                graph.edata["etype"] = etype

                # node msg
                graph.update_all(message_fn, dgl.function.mean("msg", "msg"))
                # edge msg
                graph.apply_edges(update_edges)

                self_msg = self._linear3(nfeats)
                node_feats = torch.rrelu(graph.ndata["msg"] + self_msg)
                edge_feats = self._gru(graph.edata["msg"], efeats)

            return node_feats, edge_feats


def graph_to_triplets(
    graph: dgl.DGLGraph, ent_embeds: torch.Tensor, rel_embeds: torch.Tensor
) -> torch.Tensor:

    src, dst, eid = graph.edges("all")
    subj = graph.ndata["eid"][src]
    rel = graph.edata["rid"][eid]
    obj = graph.ndata["eid"][dst]
    embed_list = [ent_embeds[subj], rel_embeds[rel], ent_embeds[obj]]
    return torch.stack(embed_list, dim=1)


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

        self._rgcn = MultiRelGraphConv(
            hidden_size, hidden_size, num_layers, dropout
        )
        self._gru = torch.nn.GRU(hidden_size, hidden_size, batch_first=True)

        self._linear1 = torch.nn.Linear(hidden_size, hidden_size)

        self.ent_embeds = torch.nn.Parameter(torch.zeros(num_ents, hidden_size))
        self.rel_embeds = torch.nn.Parameter(torch.zeros(num_rels, hidden_size))
        torch.nn.init.xavier_normal_(self.ent_embeds)
        torch.nn.init.xavier_uniform_(self.rel_embeds)

    def forward(
        self,
        hist_graphs: List[dgl.DGLGraph],
        subj: torch.Tensor,
        rel: torch.Tensor,
        obj: torch.Tensor,
    ):
        bg = dgl.batch(hist_graphs)
        total_nfeats = self._rgcn(bg, self.ent_embeds[bg.ndata["eid"]])
        hist_nfeat_list = torch.split_with_sizes(
            total_nfeats, bg.batch_num_nodes().tolist()
        )
        hist_nfeats = torch.stack(hist_nfeat_list, dim=1)
        # (num_ents, hist_len, hidden_size)
        hist_nhiddens, _ = self._gru(hist_nfeats, self.ent_embeds.unsqueeze(1))
        # hist_gfeats, _ = self._gru(hist_gfeats)
        # transformer_hist_nfeats = forwards.transformer_encoder_forward(
        #     self._transformer_encoder, hist_nfeats
        # )
        # transformer_nfeats = self._linear1(transformer_hist_nfeats[:, -1, :])

    def recursive_reasoning(self, hist_graphs: List[dgl.DGLGraph]):
        ent_embeds = self.ent_embeds
        rel_embeds = self.rel_embeds
        for graph in hist_graphs:
            node_feats = self._rgcn(graph, ent_embeds)
            pass
        pass

    def highway_reasoning(self, hist_graphs: List[dgl.DGLGraph]):
        bg = dgl.batch(hist_graphs)
        total_nfeats = self._rgcn(bg, self.ent_embeds[bg.ndata["eid"]])
        hist_nfeat_list = torch.split_with_sizes(
            total_nfeats, bg.batch_num_nodes().tolist()
        )
        hist_nfeats = torch.stack(hist_nfeat_list, dim=1)
        # (num_ents, hist_len, hidden_size)
        hist_nhiddens, _ = self._gru(hist_nfeats, self.ent_embeds.unsqueeze(1))
        pass


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask
