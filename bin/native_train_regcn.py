import dataclasses
import itertools

import accelerate
import py_helpers
import torch
import torch_helpers
from torch import optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import tkgl


@dataclasses.dataclass()
class Args:
    data_folder: str

    hist_len: int = 6

    hidden_size: int = 768
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

    device = torch.device("cuda")

    model = tkgl.models.REGCN(
        vocabs["entity"].max_index,
        vocabs["relation"].max_index,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        channels=args.channels,
        dropout=args.dropout,
    )
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = tkgl.TKGCriterion()
    # train_data = DataLoader(datasets["train"], batch_size=None)
    idx = [_ for _ in range(len(datasets["train"]))]

    batch_inputs: tkgl.datasets.QuadrupleBatchInput

    for epoch in range(2):
        epoch_tqdm = tqdm(idx, desc=f"Epoch: {epoch}")
        for x in epoch_tqdm:
            # for x in itertools.islice(epoch_tqdm, 10):
            batch_inputs = datasets["train"][x]
            batch_inputs = dataclasses.asdict(batch_inputs.to(device))
            batch_outputs = torch_helpers.nn.module_forward(
                model, **batch_inputs
            )
            losses = torch_helpers.nn.module_forward(
                criterion, **batch_inputs, **batch_outputs
            )
            loss = torch.sum(losses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()
