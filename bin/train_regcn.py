import dataclasses
import pprint

import py_helpers
import torch_helpers
from torch import optim
from torch.utils.data import DataLoader
from torch_helpers.engine import callbacks

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
    metric = tkgl.RelPredictionMetric()
    engine = torch_helpers.AccelerateEngine(
        model,
        callback_list=[
            callbacks.ModelSave(
                save_folder_path="./saved_models/regcn",
                model_template="regcn-epoch_{epoch}-mrr{val[r_mrr]:.6f}.pt",
                metric_value="+r_mrr",
                n=3,
            )
        ],
    )

    train_data = DataLoader(datasets["train"], shuffle=True, batch_size=None)
    history = engine.train(
        train_loader=train_data,
        criterion=tkgl.TKGCriterion(),
        optimizer=optim.Adam(model.parameters(), lr=args.lr),
        num_steps=len(train_data) * 10,
        metric=metric,
        val_data={
            "val": DataLoader(datasets["val"], batch_size=None),
            "test": DataLoader(datasets["test"], batch_size=None),
        },
    )
    metrics = engine.test(DataLoader(datasets["test"], batch_size=None), metric)
    pprint.pprint(metrics)


if __name__ == "__main__":
    main()
