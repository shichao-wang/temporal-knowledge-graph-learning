from typing import List

import dgl
import dgl.nn.pytorch
import tallow as tl
import torch
from torch import nn
from torch.nn import functional as tf

import tkgl.nn


class Tconv(nn.Module):
    def __init__(
        self,
        num_ents: int,
        num_rels: int,
        hist_len: int,
        hidden_size: int,
        num_kernels: int,
        num_layers: int,
        channels: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        ent_weight = tl.nn.random_init_embedding_weight(num_ents, hidden_size)
        rel_weight = tl.nn.random_init_embedding_weight(num_rels, hidden_size)
        time_weight = tl.nn.random_init_embedding_weight(hist_len, hidden_size)
        self._ent_embeds = nn.Parameter(ent_weight)
        self._rel_embeds = nn.Parameter(rel_weight)
        self._time_embeds = nn.Parameter(time_weight)

        self._tconv = TemporalConv(hidden_size, hidden_size, num_kernels)
        self._rgcn = tkgl.nn.RGCN(hidden_size, hidden_size, num_layers)
        self._gru = nn.GRUCell(hidden_size, hidden_size)
        self._rel_decoder = tkgl.nn.ConvTransR(
            hidden_size, channels, kernel_size=kernel_size, dropout=dropout
        )
        self._ent_decoder = tkgl.nn.ConvTransE(
            hidden_size, channels, kernel_size=kernel_size, dropout=dropout
        )

    def forward(
        self,
        hist_graphs: List[dgl.DGLGraph],
        subj: torch.Tensor,
        rel: torch.Tensor = None,
        obj: torch.Tensor = None,
    ):
        ent_embeds = self._ent_embeds
        rel_embeds = self._rel_embeds

        for graph, temp in zip(hist_graphs, self._time_embeds.unbind()):
            # ent_hiddens = ent_embeds
            # rel_hiddens = rel_embeds
            ent_hiddens = self._tconv(ent_embeds, temp)
            rel_hiddens = self._tconv(rel_embeds, temp)

            ent_embeds = self._rgcn(graph, ent_hiddens, rel_hiddens)
            rel_embeds = self._gru(self._rel_embeds, rel_embeds)

        ent_logit = self._ent_decoder(ent_embeds, rel_embeds, subj, rel)
        rel_logit = self._rel_decoder(ent_embeds, rel_embeds, subj, obj)

        return {"obj_logit": ent_logit, "rel_logit": rel_logit}


class TemporalConv(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, k: int):
        super().__init__()
        self._linear = nn.Linear(input_size, hidden_size)
        self._k = k
        self._pooling = nn.Linear(
            hidden_size * k - hidden_size + k, hidden_size
        )

    def forward(self, embeds: torch.Tensor, t: torch.Tensor):
        """
        Arguments:
            embeds: (num_embeds, embed_size)
            t: (hidden_size)
        Returns:
            hiddens: (num_embeds, hidden_size)
        """
        # shape: (k, 1, hidden // k)
        v = torch.reshape(self._linear(t), (self._k, 1, -1))
        # shape: (num_embeddings, k, hidden - hidden // k + 1)
        feature = tf.conv1d(torch.unsqueeze(embeds, dim=1), v)
        return self._pooling(feature.view(feature.size(0), -1))
