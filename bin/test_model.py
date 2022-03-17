import argparse
import os

import molurus
import torch
from molurus import hierdict
from tallow import evaluators

from tkgl.datasets import load_tkg_dataset
from tkgl.metrics import JointMetric
from tkgl.models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path")
    args = parser.parse_args()
    ckpt_path = args.checkpoint_path
    assert os.path.exists(ckpt_path)

    config_path = os.path.join(os.path.dirname(ckpt_path), "config.yml")
    cfg = hierdict.load(open(config_path))
    datasets, vocabs = molurus.smart_call(load_tkg_dataset, cfg["data"])

    model = build_model(
        cfg["model"],
        num_ents=len(vocabs["ent"]),
        num_rels=len(vocabs["rel"]),
    )
    evaluator = evaluators.Evaluator(datasets, JointMetric())
    results = evaluator.execute(model, torch.load(ckpt_path)["model"])
    print(results.T * 100)


if __name__ == "__main__":
    main()
