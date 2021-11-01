import dataclasses
import pprint

import py_helpers
import torch
import torch_helpers
import torchmetrics
from torch import nn, optim
from torch.nn import functional as f
from torch.utils.data import DataLoader
from torchmetrics import functional as tmf
from torchmetrics.collections import MetricCollection

import tkgl


class LinkPredictionCriterion(nn.Module):
    def forward(self, logits: torch.Tensor, rel: torch.Tensor):
        return f.cross_entropy(logits, rel, reduction="none")


class MRR(torchmetrics.MeanMetric):
    def update(self, rel: torch.Tensor, logits: torch.Tensor) -> None:
        value = tmf.retrieval_reciprocal_rank(
            logits, f.one_hot(rel, logits.size(-1))
        )
        return super().update(value)


class Hit(torchmetrics.MeanMetric):
    def __init__(self, k: int):
        super().__init__()
        self._k = k

    def update(self, rel: torch.Tensor, logits: torch.Tensor) -> None:
        value = tmf.retrieval_hit_rate(
            logits, f.one_hot(rel, logits.size(-1)), k=self._k
        )
        return super().update(value)


class LinkPredictionMetric(MetricCollection):
    def __init__(self):
        super().__init__(
            {
                "mrr": MRR(),
                "hit@1": Hit(k=1),
                "hit@3": Hit(k=3),
                "hit@10": Hit(k=10),
            }
        )

    def update(self, rel: torch.Tensor, logits: torch.Tensor) -> None:
        return super().update(rel, logits)


@dataclasses.dataclass()
class Args:
    data_folder: str

    hist_len: int = 6

    hidden_size: int = 200
    num_layers: int = 2
    kernel_size: int = 3
    channels: int = 50
    dropout: float = 0.2

    lr: float = 1e-3


def main():
    args = py_helpers.dataclass_parser.parse(Args)
    torch_helpers.seed_all()
    datasets, vocabs = tkgl.datasets.load_tkg_dataset(
        args.data_folder, args.hist_len, bidirectional=True
    )
    print(f"# entities {len(vocabs['entity'])}")
    print(f"# relations {len(vocabs['relation'])}")

    model = tkgl.models.RecurrentRGCNForLinkPrediction(
        vocabs["entity"].max_index,
        vocabs["relation"].max_index,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        channels=args.channels,
        dropout=args.dropout,
    )
    engine = torch_helpers.AccelerateEngine(
        model, metric=LinkPredictionMetric()
    )

    train_data = DataLoader(datasets["train"], shuffle=True, batch_size=None)
    history = engine.train(
        train_loader=train_data,
        criterion=LinkPredictionCriterion(),
        optimizer=optim.Adam(model.parameters(), lr=args.lr),
        num_steps=len(train_data) * 10,
        val_data={
            "Val": DataLoader(datasets["val"], batch_size=None),
            "Test": DataLoader(datasets["test"], batch_size=None),
        },
    )
    metrics = engine.test(DataLoader(datasets["test"], batch_size=None))
    pprint.pprint(metrics)


if __name__ == "__main__":
    main()
