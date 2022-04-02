from typing import List, Tuple

import dgl
import torch
import torch.nn.functional as f

from tkgl.models.hiertkgr import build_r2n_degree, group_reduce_nodes
from tkgl.models.regcn import OmegaRelGraphConv
from tkgl.models.tkgr_model import TkgrModel
from tkgl.modules.convtranse import ConvTransE
from tkgl.modules.mrgcn import MultiRelGraphConv


class REGAT(TkgrModel):
    def __init__(
        self,
        num_ents: int,
        num_rels: int,
        hidden_size: int,
        dropout: float,
        rgcn_num_layers: int,
        rgcn_num_heads: int,
        convtranse_kernel_size: int,
        convtranse_channels: int,
    ):
        super().__init__(num_ents, num_rels, hidden_size)
        self.mrgcn = MultiRelGraphConv(
            hidden_size, hidden_size, rgcn_num_heads, rgcn_num_layers, dropout
        )
        self.glinear = torch.nn.Linear(2 * hidden_size, hidden_size)
        self.relrnn = torch.nn.GRUCell(3 * hidden_size, hidden_size)
        self.convtranse = ConvTransE(
            hidden_size,
            2,
            convtranse_channels,
            convtranse_kernel_size,
            dropout,
        )

    def evolve(
        self, hist_graphs: List[dgl.DGLGraph]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ent_emb = f.normalize(self.ent_emb)
        rel_emb = self.rel_emb

        ent_emb_list = []
        rel_emb_list = []
        for graph in hist_graphs:
            node_feats = ent_emb[graph.ndata["eid"]]
            # relation evolution
            rel_in_degree, rel_out_degree = build_r2n_degree(
                graph, graph.edata["rid"], self.num_rels
            )
            nids = torch.nonzero(rel_in_degree)[:, 1]
            rel_in_emb = dgl.ops.segment_reduce(
                torch.count_nonzero(rel_in_degree, dim=1),
                ent_emb[nids],
                reducer="mean",
            )
            nids = torch.nonzero(rel_out_degree)[:, 1]
            rel_out_emb = dgl.ops.segment_reduce(
                torch.count_nonzero(rel_out_degree, dim=1),
                ent_emb[nids],
                reducer="mean",
            )
            # rel_ent_emb = group_reduce_nodes(
            #     graph, node_feats, graph.edata["rid"], num_rels=self.num_rels
            # )
            rnn_input = torch.cat(
                [self.rel_emb, rel_in_emb, rel_out_emb], dim=-1
            )
            rel_emb = f.normalize(self.relrnn(rnn_input, rel_emb))
            # entity evolution
            edge_feats = rel_emb[graph.edata["rid"]]
            neigh_feats = f.normalize(self.mrgcn(graph, node_feats, edge_feats))
            cur_ent_emb = neigh_feats[torch.argsort(graph.ndata["eid"])]
            gate_inp = torch.cat([ent_emb, cur_ent_emb], dim=-1)
            u = torch.sigmoid(self.glinear(gate_inp))
            ent_emb = f.normalize(u * cur_ent_emb + (1 - u) * ent_emb)

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
    ):
        """

        Arguments:
            snapshot: [his_len]
            triplets: (num_triplets, 3)

        Returns:
            logits: (num_triplets, num_entities)
        """
        hist_ent_emb, hist_rel_emb = self.evolve(hist_graphs)
        ent_emb = hist_ent_emb[-1]
        rel_emb = hist_rel_emb[-1]

        obj_inp = torch.stack([ent_emb[subj], rel_emb[rel]], dim=1)
        obj_logit = self.convtranse(obj_inp) @ ent_emb.t()
        return {
            "obj_logit": obj_logit,
            "hist_ent_emb": hist_ent_emb,
            "hist_rel_emb": hist_rel_emb,
        }
