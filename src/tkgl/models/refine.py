from typing import List

import dgl
import torch
from tallow.nn import forwards
from torch import nn

# from tkgl.nn import RGCN
from tkgl.nn.decoders import ConvTransBackbone, ConvTransE, ConvTransR
from tkgl.nn.rgcn import RGCN

from .regcn import EvoRGCN

# class RefineConvTransE(torch.nn.Module):
#     def __init__(
#         self,
#         hidden_size: int,
#         channels: int,
#         kernel_size: int,
#         dropout: float,
#     ):
#         super().__init__()
#         self._backbone1 = ConvTransBackbone(
#             hidden_size,
#             in_channels=2,
#             out_channels=channels,
#             kernel_size=kernel_size,
#             dropout=dropout,
#         )
#         self._backbone2 = ConvTransBackbone(
#             hidden_size,
#             in_channels=3,
#             out_channels=channels,
#             kernel_size=kernel_size,
#             dropout=dropout,
#         )

#     def forward(
#         self,
#         nodes: torch.Tensor,
#         edges: torch.Tensor,
#         subjs: torch.Tensor,
#         rels: torch.Tensor,
#     ):
#         """
#         Arguments:
#             nodes: (num_nodes, hidden_size)
#             edges: (num_edges, hidden_size)
#             heads: (num_triplets,)
#             relations: (num_triplets,)
#         Return:

#         """
#         # (num_triplets, hidden_size)
#         subj_embeds = nodes[subjs]
#         rel_embeds = edges[rels]
#         # (num_triplets, 2, hidden_size)
#         x = torch.stack([subj_embeds, rel_embeds], dim=1)
#         ori_embeds: torch.Tensor = self._backbone1(x)
#         x = torch.stack([ori_embeds, subj_embeds, rel_embeds], dim=1)
#         re_embeds = self._backbone2(x)
#         return {"ori": ori_embeds @ nodes.t(), "re": re_embeds @ nodes.t()}


# class RefineConvTransR(torch.nn.Module):
#     def __init__(
#         self,
#         hidden_size: int,
#         channels: int,
#         kernel_size: int,
#         dropout: float,
#     ):
#         super().__init__()
#         self._backbone1 = ConvTransBackbone(
#             hidden_size,
#             in_channels=2,
#             out_channels=channels,
#             kernel_size=kernel_size,
#             dropout=dropout,
#         )
#         self._backbone2 = ConvTransBackbone(
#             hidden_size,
#             in_channels=3,
#             out_channels=channels,
#             kernel_size=kernel_size,
#             dropout=dropout,
#         )

#     def forward(
#         self,
#         nodes: torch.Tensor,
#         edges: torch.Tensor,
#         subjs: torch.Tensor,
#         objs: torch.Tensor,
#     ):
#         """ """
#         # (num_triplets, hidden_size)
#         subj_embeds = nodes[subjs]
#         obj_embeds = nodes[objs]
#         # (num_triplets, 2, hidden_size)
#         x = torch.stack([subj_embeds, obj_embeds], dim=1)
#         ori_embeds: torch.Tensor = self._backbone1(x)
#         x = torch.stack([ori_embeds, subj_embeds, obj_embeds], dim=1)
#         re_embeds = self._backbone2(x)
#         return {"ori": ori_embeds @ edges.t(), "re": re_embeds @ edges.t()}


class Refine(torch.nn.Module):
    def __init__(
        self,
        num_ents: int,
        num_rels: int,
        hidden_size: int,
        num_layers: int,
        channels: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()

        self.ent_embeds = torch.nn.Parameter(torch.zeros(num_ents, hidden_size))
        self.rel_embeds = torch.nn.Parameter(torch.zeros(num_rels, hidden_size))
        torch.nn.init.normal_(self.ent_embeds)
        torch.nn.init.xavier_uniform_(self.rel_embeds)

        self._rgcn = RGCN(hidden_size, hidden_size, num_layers)
        self._gru = nn.GRU(
            hidden_size,
            hidden_size,
            batch_first=True,
            dropout=dropout,
        )
        self._evo_rgcn = EvoRGCN(hidden_size, hidden_size, num_layers, dropout)
        self._linear = nn.Linear(hidden_size, hidden_size)
        self._rel_gru = nn.GRUCell(2 * hidden_size, hidden_size)
        self._ent_gru = nn.GRUCell(hidden_size, hidden_size)

        self._rel_decoder = ConvTransR(
            hidden_size, channels, kernel_size, dropout
        )
        self._obj_decoder = ConvTransE(
            hidden_size, channels, kernel_size, dropout
        )
        # self._obj_linear = nn.Sequential(
        #     nn.Linear(3 * hidden_size, hidden_size),
        #     nn.Tanh(),
        #     nn.Linear(hidden_size, num_ents),
        # )
        # self._rel_linear = nn.Sequential(
        #     nn.Linear(3 * hidden_size, hidden_size),
        #     nn.Tanh(),
        #     nn.Linear(hidden_size, num_rels),
        # )

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
        bg = dgl.batch(hist_graphs)
        # (hist_len * num_nodes, h)
        bnfeat = self._rgcn(bg, self.ent_embeds, self.rel_embeds)
        # (hist_len, h)
        graph_embeds: torch.Tensor = dgl.ops.segment_reduce(
            bg.batch_num_nodes(), bnfeat, "max"
        )
        graph_embeds, _ = forwards.rnn_forward(
            self._gru, graph_embeds.unsqueeze(0)
        )

        ent_embeds = self.ent_embeds
        rel_embeds = self.rel_embeds

        for graph, graph_embed in zip(hist_graphs, graph_embeds):
            # ent evolution
            e_ent_embeds = self._evo_rgcn(graph, ent_embeds, rel_embeds)
            u = torch.sigmoid(self._linear(ent_embeds))
            n_ent_embeds = ent_embeds + u * (e_ent_embeds - ent_embeds)
            # rel evolution
            e_rel_embeds = self.rel_embeds
            e_rel_embeds = torch.cat([self.rel_embeds, e_rel_embeds], dim=-1)
            n_rel_embeds = self._rel_gru(e_rel_embeds, rel_embeds)

            ent_embeds = n_ent_embeds
            rel_embeds = n_rel_embeds

        rel_logit = self._rel_decoder(ent_embeds, rel_embeds, subj, obj)
        obj_logit = self._obj_decoder(ent_embeds, rel_embeds, subj, rel)

        # graph_embed = _.repeat(len(subj), 1)
        # rel_logit = self._rel_linear(
        #     torch.cat([graph_embed, ent_embeds[subj], ent_embeds[obj]], dim=-1)
        # )
        # obj_logit = self._obj_linear(
        #     torch.cat([graph_embed, ent_embeds[subj], rel_embeds[rel]], dim=-1)
        # )

        return {"rel_logit": rel_logit, "obj_logit": obj_logit}
