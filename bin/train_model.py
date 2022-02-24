import logging
import os
from typing import Dict

import torch
from molurus import config_dict
from tallow import evaluators, trainers
from tallow.data import vocabs
from torch import nn, optim

import tkgl
from tkgl.metrics import EntMRR, JointMetric

logger = logging.getLogger(__name__)


def validate_module_parameters(m1: nn.Module, m2: nn.Module):
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        p1 = p1.cpu()
        p2 = p2.cpu()
        if not torch.allclose(p1, p2):
            logger.info(f"Tensor mismatch: {p1} vs {p2}")


def validate_state_dicts(m1: Dict, m2: Dict):
    if len(m1) != len(m2):
        logger.info(f"Length mismatch: {len(m1)}, {len(m2)}")
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(m1.keys())).startswith("module"):
        m1 = {k[len("module") + 1 :]: v for k, v in m1.items()}

    if next(iter(m2.keys())).startswith("module"):
        m2 = {k[len("module") + 1 :]: v for k, v in m2.items()}

    for ((k_1, v_1), (k_2, v_2)) in zip(m1.items(), m2.items()):
        if k_1 != k_2:
            logger.info(f"Key mismatch: {k_1} vs {k_2}")
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2):
            logger.info(f"Tensor mismatch: {v_1} vs {v_2}")
            # return False


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
        )
        criterion = tkgl.models.criterions.JointLoss()
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
        criterion = tkgl.models.criterions.JointLoss()
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
        criterion = tkgl.models.criterions.RefineLoss()
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
        return optim.Adam(m.parameters(), lr=cfg["lr"])

    val_metric = tkgl.metrics.EntMRR()

    ckpt = trainers.hooks.CheckpointHook(
        cfg["save_folder_path"], reload_on_begin=True
    )
    earlystop = trainers.hooks.EarlyStopHook(
        cfg["save_folder_path"], patient=cfg["patient"]
    )
    trainer = trainers.SupervisedTrainer(
        model,
        criterion=criterion,
        init_optim=_init_optim,
        metric=val_metric,
        _hooks=[ckpt, earlystop],
    )
    trainer.execute(datasets["train"], eval_data=datasets["val"])

    metric = JointMetric()
    evaluator = evaluators.trainer_evaluator(trainer, metric)
    evaluator.model.load_state_dict(
        torch.load(earlystop.best_model_path)["model"]
    )
    metric_dataframe = evaluator.execute(datasets)
    print(metric_dataframe)


if __name__ == "__main__":
    main()
