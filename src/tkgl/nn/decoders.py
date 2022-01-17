import torch
from torch import nn


class ConvTransBackbone(nn.Module):
    """
    The backbone module for ConvTransE and ConvTransR.
    The Conv1d here is not a common way to process NLP features.
    """

    def __init__(
        self,
        hidden_size: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self._bn = nn.BatchNorm1d(in_channels)
        self._dp = nn.Dropout(dropout)
        self._conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self._bn1 = nn.BatchNorm1d(out_channels)
        self._dp1 = nn.Dropout(dropout)
        self._linear = nn.Linear(hidden_size * out_channels, hidden_size)
        self._bn2 = nn.BatchNorm1d(hidden_size)
        self._dp2 = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
    ):
        """
        Arguments:
            inputs: (num_triplets, in_channels, hidden_size)
        Return:
            output: (num_triplets, hidden_size)
        """
        num_triplets = inputs.size(0)
        # (num_triplets, out_channels, embed_size)
        feature_maps = self._conv1(self._dp(self._bn(inputs)))
        _ = self._dp1(self._bn1(feature_maps).relu())
        hidden = self._linear(_.view(num_triplets, -1))
        # (num_triplets, 1, embed_size)
        embedding = self._bn2(self._dp2(hidden)).relu()
        return embedding
        # (num_triplets, embed_size)


class ConvTransE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        channels: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self._backbone = ConvTransBackbone(
            hidden_size,
            in_channels=2,
            out_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        heads: torch.Tensor,
        relations: torch.Tensor,
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
        head_embedding = nodes[heads]
        rel_embedding = nodes[relations]
        # (num_triplets, 2, hidden_size)
        x = torch.stack([head_embedding, rel_embedding], dim=1)
        embedding = self._backbone(x)
        return torch.sigmoid(embedding @ nodes.t())


class ConvTransR(nn.Module):
    def __init__(
        self, hidden_size: int, channels: int, kernel_size: int, dropout: float
    ):
        super().__init__()
        self._backbone = ConvTransBackbone(
            hidden_size,
            in_channels=2,
            out_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        heads: torch.Tensor,
        tails: torch.Tensor,
    ):
        """
        Arguments:
            nodes: (num_nodes, embed_size)
            edges: (num_edges, embed_size)
            heads: (num_triplets,)
            tail: (num_triplets,)
        """
        # (num_triplets, embed_size)
        head_embedding = nodes[heads]
        tail_embedding = nodes[tails]
        # (num_triplets, 2, embed_size)
        x = torch.stack([head_embedding, tail_embedding], dim=1)
        embedding = self._backbone(x)
        return torch.sigmoid(embedding @ edges.t())
