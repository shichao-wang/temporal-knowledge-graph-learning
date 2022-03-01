import logging
import os
from typing import Dict

import torch
from molurus import config_dict
from tallow import evaluators, trainers
from tallow.data import vocabs
from torch import nn, optim

import tkgl

logger = logging.getLogger(__name__)


def build_model(cfg, tknzs: Dict[str, vocabs.Vocab]):
    if cfg["arch"] == "regcn":
        model = tkgl.models.REGCN(
            len(tknzs["ent"]),
            len(tknzs["rel"]),
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            kernel_size=cfg["kernel_size"],
            channels=cfg["channels"],
            dropout=cfg["dropout"],
            norm_embeds=cfg["norm_embeds"],
        )
        criterion = tkgl.models.criterions.JointLoss(balance=cfg["alpha"])
    elif cfg["arch"] == "tconv":
        model = tkgl.models.Tconv(
            len(tknzs["ent"]),
            len(tknzs["rel"]),
            hist_len=cfg["hist_len"],
            hidden_size=cfg["hidden_size"],
            num_kernels=cfg["num_kernels"],
            num_layers=cfg["num_layers"],
            kernel_size=cfg["kernel_size"],
            channels=cfg["channels"],
            dropout=cfg["dropout"],
        )
        criterion = tkgl.models.criterions.JointLoss(balance=cfg["alpha"])
    elif cfg["arch"] == "refine":
        model = tkgl.models.Refine(
            len(tknzs["ent"]),
            len(tknzs["rel"]),
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            channels=cfg["channels"],
            kernel_size=cfg["kernel_size"],
            dropout=cfg["dropout"],
        )
        criterion = tkgl.models.criterions.JointLoss(balance=cfg["alpha"])
        # criterion = tkgl.models.criterions.RefineLoss(
        #     alpha=cfg["alpha"], beta=cfg["beta"]
        # )
    else:
        raise ValueError()
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Criterion: {criterion.__class__.__name__}")
    return model, criterion


def load_tkg_data(cfg):
    datasets, vocabs = tkgl.datasets.load_tkg_dataset(
        os.path.join(cfg["data_folder"], cfg["dataset"]),
        cfg["hist_len"],
        bidirectional=cfg["bidirectional"],
        different_unknowns=cfg["different_unknowns"],
        complement_val_and_test=cfg["complement_val_and_test"],
        shuffle=cfg["shuffle"],
    )
    return datasets, vocabs


def main():
    cfg = config_dict.parse()
    assert "save_folder_path" in cfg
    os.makedirs(cfg["save_folder_path"], exist_ok=True)
    with open(os.path.join(cfg["save_folder_path"], "config.yml"), "w") as fp:
        config_dict.dump(cfg, fp)
    datasets, vocabs = load_tkg_data(cfg["data"])

    num_entities = len(vocabs["ent"])
    print(f"# entities {num_entities}")
    num_relations = len(vocabs["rel"])
    print(f"# relations {num_relations}")
    num_edges = datasets["train"]
    print(f"# edges {num_edges}")

    model, criterion = build_model(cfg["model"], vocabs)

    def _init_optim(m: nn.Module):
        optimizer = optim.Adam(
            m.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        clip_norm = None
        if cfg["grad_norm"]:
            clip_norm = lambda: nn.utils.clip_grad.clip_grad_norm_(
                m.parameters(), cfg["grad_norm"]
            )

        return (optimizer, clip_norm)

    val_metric = tkgl.metrics.EntMRR()

    hooks = []
    ckpt = trainers.hooks.CheckpointHook(
        cfg["save_folder_path"], reload_on_begin=True
    )
    hooks.append(ckpt)
    earlystop = trainers.hooks.EarlyStopHook(
        cfg["save_folder_path"], patient=cfg["patient"]
    )
    hooks.append(earlystop)
    if cfg["val_test"]:
        val_hook = trainers.hooks.EvaluateHook({"test": datasets["test"]})
        hooks.append(val_hook)

    trainer = trainers.SupervisedTrainer(
        model,
        criterion=criterion,
        init_optim=_init_optim,
        metric=val_metric,
        _hooks=hooks,
    )
    trainer.execute(datasets["train"], eval_data=datasets["val"])

    full_metric = tkgl.metrics.JointMetric()
    evaluator = evaluators.trainer_evaluator(trainer, full_metric)
    evaluator.model.load_state_dict(
        torch.load(earlystop.best_model_path)["model"]
    )
    metric_dataframe = evaluator.execute(datasets)
    print(metric_dataframe.T * 100)


if __name__ == "__main__":
    main()
