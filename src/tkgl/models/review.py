from typing import List

import dgl
import torch

from .tkgr_model import TkgrModel


class ReviewNet(TkgrModel):
    def __init__(self, num_ents: int, num_rels: int, hidden_size: int):
        super().__init__()
        self.ent_emb = torch.nn.Parameter(torch.empty(num_ents, hidden_size))
        self.rel_emb = torch.nn.Parameter(torch.empty(num_rels, hidden_size))
        torch.nn.init.xavier_normal_(self.ent_emb)
        torch.nn.init.xavier_normal_(self.rel_emb)

    def forward(
        self,
        hist_graphs: List[dgl.DGLGraph],
        subj: torch.Tensor,
        rel: torch.Tensor,
        obj: torch.Tensor,
    ):
        bg = dgl.batch(hist_graphs)
        nfeats, efeats = self._rgcn(
            bg, self.ent_emb[bg.ndata["eid"]], self.rel_emb[bg.edata["rid"]]
        )
