import torch


class ConvTransE(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self._bn1 = torch.nn.BatchNorm1d(in_channels)
        self._dp1 = torch.nn.Dropout(dropout)
        self._conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self._bn2 = torch.nn.BatchNorm1d(out_channels)
        self._dp2 = torch.nn.Dropout(dropout)
        self._linear = torch.nn.Linear(hidden_size * out_channels, hidden_size)
        self._bn3 = torch.nn.BatchNorm1d(hidden_size)
        self._dp3 = torch.nn.Dropout(dropout)

    def forward(self, tuples: torch.Tensor, emb: torch.Tensor):
        """
        Arguments:
            inputs: (num_triplets, in_channels, hidden_size)
            emb: (num_classes, hidden_size)
        Return:
            output: (num_triplets, hidden_size)
        """
        num_inputs = tuples.size(0)
        # (num_triplets, out_channels, embed_size)
        feature_maps = self._conv(self._dp1(self._bn1(tuples)))
        _ = self._dp2(self._bn2(feature_maps).relu())
        # (num_triplets, embed_size)
        hidden = self._linear(_.view(num_inputs, -1))
        # (num_triplets, embed_size)
        score = self._bn3(self._dp3(hidden)).relu()
        return score @ emb.t()
