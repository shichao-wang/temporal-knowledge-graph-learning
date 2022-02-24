from typing import List

import dgl
import torch

from tkgl.nn.decoders import ConvTransBackbone

from .regcn import EvolutionUnit


class RefineConvTransE(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        channels: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self._backbone1 = ConvTransBackbone(
            hidden_size,
            in_channels=2,
            out_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self._backbone2 = ConvTransBackbone(
            hidden_size,
            in_channels=3,
            out_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        subjs: torch.Tensor,
        rels: torch.Tensor,
    ):
        """
        Arguments:
            nodes: (num_nodes, hidden_size)
            edges: (num_edges, hidden_size)
            heads: (num_triplets,)
            relations: (num_triplets,)
        Return:

        """
        # (num_triplets, hidden_size)
        subj_embeds = nodes[subjs]
        rel_embeds = edges[rels]
        # (num_triplets, 2, hidden_size)
        x = torch.stack([subj_embeds, rel_embeds], dim=1)
        ori_embeds: torch.Tensor = self._backbone1(x)
        x = torch.stack([ori_embeds, subj_embeds, rel_embeds], dim=1)
        re_embeds = self._backbone2(x)
        return {"ori": ori_embeds @ nodes.t(), "re": re_embeds @ nodes.t()}


class RefineConvTransR(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        channels: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self._backbone1 = ConvTransBackbone(
            hidden_size,
            in_channels=2,
            out_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self._backbone2 = ConvTransBackbone(
            hidden_size,
            in_channels=3,
            out_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        subjs: torch.Tensor,
        objs: torch.Tensor,
    ):
        """ """
        # (num_triplets, hidden_size)
        subj_embeds = nodes[subjs]
        obj_embeds = nodes[objs]
        # (num_triplets, 2, hidden_size)
        x = torch.stack([subj_embeds, obj_embeds], dim=1)
        ori_embeds: torch.Tensor = self._backbone1(x)
        x = torch.stack([ori_embeds, subj_embeds, obj_embeds], dim=1)
        re_embeds = self._backbone2(x)
        return {"ori": ori_embeds @ edges.t(), "re": re_embeds @ edges.t()}


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

        self._ent_embeds = torch.nn.Parameter(
            torch.zeros(num_ents, hidden_size)
        )
        self._rel_embeds = torch.nn.Parameter(
            torch.zeros(num_rels, hidden_size)
        )
        torch.nn.init.normal_(self._ent_embeds)
        torch.nn.init.xavier_uniform_(self._rel_embeds)

        self._evolution = EvolutionUnit(hidden_size, hidden_size, num_layers)
        self._rel_decoder = RefineConvTransR(
            hidden_size, channels, kernel_size, dropout
        )
        self._ent_decoder = RefineConvTransE(
            hidden_size, channels, kernel_size, dropout
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
        rel_outputs = self._rel_decoder(ent_embeds, rel_embeds, subj, obj)
        obj_outputs = self._ent_decoder(ent_embeds, rel_embeds, subj, rel)

        return {
            "obj_logit_ori": obj_outputs["ori"],
            "rel_logit_ori": rel_outputs["ori"],
            "obj_logit": obj_outputs["re"],
            "rel_logit": rel_outputs["re"],
        }
