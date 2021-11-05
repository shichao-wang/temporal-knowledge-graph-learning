import dataclasses
import pprint

import py_helpers
import torch_helpers
from torch import optim
from torch.utils.data import DataLoader

import tkgl


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
    torch_helpers.seed_all(1234)
    datasets, vocabs = tkgl.datasets.load_tkg_dataset(
        args.data_folder, args.hist_len, bidirectional=True
    )
    print(f"# entities {len(vocabs['entity'])}")
    print(f"# relations {len(vocabs['relation'])}")

    model = tkgl.models.REGCN(
        vocabs["entity"].max_index,
        vocabs["relation"].max_index,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        channels=args.channels,
        dropout=args.dropout,
    )
    engine = torch_helpers.AccelerateEngine(
        model, metric=tkgl.RelPredictionMetric()
    )

    train_data = DataLoader(datasets["train"], shuffle=True, batch_size=None)
    history = engine.train(
        train_loader=train_data,
        criterion=tkgl.TKGCriterion(),
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
