import logging
import os

import molurus
from molurus import hierdict
from tallow.common.seeds import seed_all
from tallow.data.datasets import Dataset
from tallow.evaluators import Evaluator
from tallow.metrics import TorchMetric
from tallow.trainers import Trainer

from tkgl.datasets import load_tkg_dataset
from tkgl.metrics import EntMetric
from tkgl.models.tkgr_model import TkgrModel

logger = logging.getLogger(__name__)


def train_model(
    save_folder_path: str,
    model: TkgrModel,
    train_data: Dataset,
    val_data: Dataset,
    metric: TorchMetric,
    tr_cfg: hierdict.HierDict,
):
    print(tr_cfg)
    criterion = molurus.smart_instantiate(tr_cfg["criterion"])
    optimizer = molurus.smart_instantiate(
        tr_cfg["optim"],
        params=(p for p in model.parameters() if p.requires_grad),
    )
    trainer = Trainer(
        save_folder_path,
        earlystop_monitor="e_mrr",
        earlystop_patient=tr_cfg["patient"],
        grad_clip_norm=tr_cfg["grad_clip_norm"],
    )
    state_dict = trainer.execute(
        model, train_data, criterion, optimizer, val_data, metric
    )
    return state_dict


def validate_and_dump_config(
    save_folder_path: str, cfg: hierdict.HierDict, cfg_file: str = "config.yml"
):
    os.makedirs(save_folder_path, exist_ok=True)
    cfg_path = os.path.join(save_folder_path, cfg_file)
    if not os.path.exists(cfg_path):
        with open(cfg_path, "w") as fp:
            hierdict.dump(cfg, fp)

    orig_cfg = hierdict.load(open(cfg_path))
    assert orig_cfg == cfg


def train(save_folder_path: str, cfg: hierdict.HierDict) -> float:
    cfg.setdefault("seed", 1234)
    validate_and_dump_config(save_folder_path, cfg)
    seed_all(cfg["seed"])
    datasets, vocabs = load_tkg_dataset(**cfg["data"])
    model = molurus.smart_instantiate(
        cfg["model"], num_ents=len(vocabs["ent"]), num_rels=len(vocabs["rel"])
    )
    metric = EntMetric()
    best_state_dict = train_model(
        save_folder_path,
        model,
        datasets["train"],
        val_data=datasets["valid"],
        metric=metric,
        tr_cfg=cfg["training"],
    )

    evaluator = Evaluator(datasets, metric)
    dataframe = evaluator.execute(model, best_state_dict["model"])
    print(dataframe * 100)
    return dataframe["e_mrr"]["valid"]


def main():
    parser = hierdict.ArgumentParser()
    parser.add_argument("--save-folder-path")
    args = parser.parse_args()
    cfg = hierdict.parse_args(args.cfg, args.overrides)
    train(args.save_folder_path, cfg)


if __name__ == "__main__":
    main()
