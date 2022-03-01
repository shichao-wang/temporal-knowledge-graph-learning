import argparse
import os
import pprint

import torch
from molurus import config_dict
from tallow import evaluators
from train_model import build_model, load_tkg_data

import tkgl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path")
    args = parser.parse_args()
    ckpt_path = args.checkpoint_path
    assert os.path.exists(ckpt_path)

    config_path = os.path.join(os.path.dirname(ckpt_path), "config.yml")
    cfg = config_dict.parse_file(config_path)
    datasets, vocabs = load_tkg_data(cfg["data"])

    model, criterion = build_model(cfg["model"], vocabs)
    model.load_state_dict(torch.load(ckpt_path)["model"])
    metric = tkgl.metrics.JointMetric()
    evaluator = evaluators.Evaluator(model, metric)
    results = evaluator.execute(datasets)
    print(results.T * 100)


if __name__ == "__main__":
    main()
