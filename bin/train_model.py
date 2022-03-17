import os

import molurus
from molurus import hierdict
from tallow import backends
from tallow.backends.grad_clipper import grad_clipper
from tallow.common.seeds import seed_all
from tallow.data.datasets import Dataset
from tallow.evaluators import Evaluator
from tallow.trainers import Trainer
from tallow.trainers.hooks import EarlyStopHook, HookManager

from tkgl import models
from tkgl.datasets import load_tkg_dataset
from tkgl.metrics import EntMRR, JointMetric
from tkgl.models.tkgr_model import TkgrModel


def train_model(
    save_folder_path: str,
    model: TkgrModel,
    train_data: Dataset,
    val_data: Dataset,
    tr_cfg: hierdict.HierDict,
):
    seed_all(tr_cfg.get("seed", 1234))
    os.makedirs(save_folder_path, exist_ok=True)
    val_metric = EntMRR()
    backend = molurus.smart_call(backends.auto_select, tr_cfg)

    criterion = molurus.smart_call(model.build_criterion, tr_cfg)
    optimizer = molurus.smart_instaniate(
        "torch.optim.Adam",
        tr_cfg,
        params=(p for p in model.parameters() if p.requires_grad),
    )
    clipper = molurus.smart_call(grad_clipper, tr_cfg)
    earlystop = EarlyStopHook(
        save_folder_path,
        val_data,
        val_metric,
        tr_cfg["patient"],
        backend=backend,
    )
    hook_mgr = HookManager(earlystop)
    trainer = Trainer(backend, clipper, hook_mgr=hook_mgr)
    last_state_dict = trainer.execute(model, train_data, criterion, optimizer)
    best_state_dict = earlystop._disk_mgr.load()
    return best_state_dict


def validate_and_dump_config(
    save_folder_path: str, cfg: hierdict.HierDict, cfg_file: str = "config.yml"
):
    cfg_path = os.path.join(save_folder_path, cfg_file)
    if not os.path.exists(cfg_path):
        with open(cfg_path, "w") as fp:
            hierdict.dump(cfg, fp)

    orig_cfg = hierdict.load(open(cfg_path))
    assert orig_cfg == cfg


def main():
    parser = hierdict.ArgumentParser()
    parser.add_argument("--save-folder-path")
    args = parser.parse_args()
    cfg = hierdict.parse_args(args.cfg, args.overrides)

    os.makedirs(args.save_folder_path, exist_ok=True)
    validate_and_dump_config(args.save_folder_path, cfg)

    datasets, vocabs = molurus.smart_call(load_tkg_dataset, cfg["data"])
    model = models.build_model(
        cfg["model"], num_ents=len(vocabs["ent"]), num_rels=len(vocabs["rel"])
    )
    best_state_dict = train_model(
        args.save_folder_path,
        model,
        datasets["train"],
        val_data=datasets["val"],
        tr_cfg=cfg["training"],
    )

    evaluator = Evaluator(datasets, JointMetric())
    dataframe = evaluator.execute(model, best_state_dict["model"])
    print(dataframe.T * 100)


if __name__ == "__main__":
    main()
