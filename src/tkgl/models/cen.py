from typing import List, Tuple

import dgl
import torch
from torch.nn import functional as f

from tkgl.modules.convtranse import ConvTransE

from .regcn import OmegaRelGraphConv
from .tkgr_model import TkgrModel


class ComplexEvoNet(TkgrModel):
    def __init__(
        self,
        num_ents: int,
        num_rels: int,
        hidden_size: int,
        dropout: float,
        norm_embeds: bool,
        rgcn_num_layers: int,
        rgcn_self_loop: bool,
        convtranse_kernel_size: int,
        convtranse_channels: int,
    ):
        super().__init__(num_ents, num_rels, hidden_size)
        self.rgcn = OmegaRelGraphConv(
            hidden_size, hidden_size, rgcn_num_layers, rgcn_self_loop, dropout
        )
        self.glinear = torch.nn.Linear(hidden_size, hidden_size)
        self.obj_convtranse = ConvTransE(
            hidden_size,
            2,
            convtranse_channels,
            convtranse_kernel_size,
            dropout,
        )
        self._norm_embeds = norm_embeds

    def evolve(
        self, hist_graphs: List[dgl.DGLGraph]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ent_emb = self._origin_or_norm(self.ent_emb)

        hist_ent_emb = []
        for graph in hist_graphs:
            node_feats = ent_emb[graph.ndata["eid"]]
            # entity evolution
            edge_feats = self.rel_emb[graph.edata["rid"]]
            neigh_feats = self.rgcn(graph, node_feats, edge_feats)
            neigh_feats = self._origin_or_norm(neigh_feats)
            ent_neigh_emb = neigh_feats[torch.argsort(graph.ndata["eid"])]
            u = torch.sigmoid(self.glinear(ent_emb))
            ent_emb = f.normalize(u * ent_neigh_emb + (1 - u) * ent_emb)

            hist_ent_emb.append(ent_emb)

        return torch.stack(hist_ent_emb)

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
        hist_len = len(hist_graphs)
        hist_obj_logit_list = []
        for i in range(hist_len):
            hist_ent_emb = self.evolve(hist_graphs[i:])
            ent_emb = hist_ent_emb[-1]
            obj_inp = torch.stack([ent_emb[subj], self.rel_emb[rel]], dim=1)
            obj_logit = self.obj_convtranse(obj_inp) @ ent_emb.t()
            hist_obj_logit_list.append(obj_logit)

        hist_obj_logit = torch.stack(hist_obj_logit_list)
        obj_logit = hist_obj_logit.softmax(dim=-1).sum(dim=0)

        return {
            "hist_obj_logit": hist_obj_logit,
            # "hist_ent_emb": hist_ent_emb,
            "obj_logit": obj_logit,  # used for prediction
        }

    def _origin_or_norm(self, tensor: torch.Tensor):
        if self._norm_embeds:
            return f.normalize(tensor)
        return tensor


class HistEntLoss(torch.nn.Module):
    def forward(self, hist_obj_logit: torch.Tensor, obj: torch.Tensor):
        loss = obj.new_tensor(0.0)
        for obj_logit in hist_obj_logit.unbind(0):
            loss = loss + f.cross_entropy(obj_logit, obj)
        return loss
