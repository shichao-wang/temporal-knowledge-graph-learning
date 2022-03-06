from typing import List

import dgl
import torch
from tallow.nn import forwards

from tkgl.models.evokg import RGCN


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

        self._rgcn = RGCN(
            hidden_size, hidden_size, num_rels, num_layers, dropout
        )
        self._gru = torch.nn.GRU(hidden_size, hidden_size, batch_first=True)
        self._transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                hidden_size, num_heads, dropout=dropout
            ),
            num_layers,
        )
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
