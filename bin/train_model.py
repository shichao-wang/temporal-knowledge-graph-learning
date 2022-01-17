import dataclasses
import pprint

import py_helpers
import torch
import torch_helpers as th
from torch import optim
from torch_helpers.engine import callbacks

import tkgl


@dataclasses.dataclass()
class Args:
    data_folder: str

    model: str
    save_model: bool = False

    seed: int = 0
    hist_len: int = 6

    hidden_size: int = 200
    num_kernels: int = 2
    num_layers: int = 2
    kernel_size: int = 3
    channels: int = 50
    dropout: float = 0.2

    lr: float = 1e-3
    num_epochs: int = 10


def main():
    args = py_helpers.dataclass_parser.parse(Args)
    pprint.pprint(dataclasses.asdict(args))
    th.seed_all(args.seed)
    datasets, vocabs = tkgl.datasets.load_tkg_dataset(
        args.data_folder, args.hist_len, bidirectional=True
    )
    print(f"# entities {vocabs['ent'].vocab_size}")
    print(f"# relations {vocabs['rel'].vocab_size}")

    if args.model == "tconv":
        model = tkgl.models.Tconv(
            vocabs["ent"].vocab_size,
            vocabs["rel"].vocab_size,
            hist_len=args.hist_len,
            hidden_size=args.hidden_size,
            num_kernels=args.num_kernels,
            num_layers=args.num_layers,
            kernel_size=args.kernel_size,
            channels=args.channels,
            dropout=args.dropout,
        )
    elif args.model == "regcn":
        model = tkgl.models.REGCN(
            vocabs["ent"].vocab_size,
            vocabs["rel"].vocab_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            kernel_size=args.kernel_size,
            channels=args.channels,
            dropout=args.dropout,
        )
    else:
        raise ValueError()
    metric = tkgl.metrics.JointMetric()
    engine = th.NativeEngine(model, "cuda")
    model_save = callbacks.ModelSave(
        save_folder_path=f"./saved_models/{args.model}",
        model_template="epoch-{epoch}_mrr-{val[e_mrr]:.6f}-{test[e_mrr]:.6f}-{test[r_mrr]:.6f}.pt",
        metric_value="+{val[e_mrr]}",
        n=3,
    )
    training_callbacks = []
    if args.save_model:
        training_callbacks.append(model_save)

    train_data = datasets.pop("train").shuffle()
    history = engine.train(
        train_data,
        criterion=tkgl.nn.JointLoss(),
        optimizer=optim.Adam(model.parameters(), lr=args.lr),
        num_steps=len(train_data) * args.num_epochs,
        metric=metric,
        val_data=datasets,
        callback_list=training_callbacks,
    )

    if args.save_model:
        best_model_path = model_save.get_best_model_path()
        model.load_state_dict(torch.load(best_model_path))
        test_engine = th.NativeEngine(model, "cuda")
        metrics = test_engine.test(datasets["test"], metric)
        pprint.pprint(metrics)


if __name__ == "__main__":
    main()
