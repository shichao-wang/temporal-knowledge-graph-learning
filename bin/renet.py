import dataclasses
from typing import List

import dgl.data
import py_helpers
import torch
import torch_helpers
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from torch_helpers import MetricModule
from torch_helpers.nn import Embedding

import tkgl


@dataclasses.dataclass()
class Args:
    data_folder: str


class TKGMetric(MetricModule):
    def __init__(self) -> None:
        pass


def main():
    args = py_helpers.dataclass_parser.parse(Args)
    datasets, vocabs = tkgl.datasets.load_tkg_dataset(args.data_folder)
    # embeddings = {}
    # model = tkgl.models.RecurrentRGCNForLinkPrediction(
    #     embeddings["entity"],
    #     embeddings["relation"],
    #     embeddings["time"],
    #     hidden_size=args.hidden_size,
    # )
    # engine = torch_helpers.AccelerateEngine(model, metric=TKGMetric())
    # history = engine.train(
    #     train_loader=DataLoader(
    #         datasets["train"],
    #         batch_size=args.batch_size,
    #         shuffle=True,
    #         pin_memory=True,
    #     ),
    #     criterion=nn.CrossEntropyLoss(),
    #     optimizer=optim.Adam(model.parameters(), lr=args.lr),
    #     num_steps=10000,
    #     val_data=DataLoader(
    #         datasets["val"],
    #         batch_size=args.batch_size,
    #         pin_memory=True,
    #     ),
    # )
    # engine.test(
    #     DataLoader(
    #         datasets["test"],
    #         batch_size=args.batch_size,
    #         shuffle=False,
    #         pin_memory=True,
    #     )
    # )


if __name__ == "__main__":
    main()
