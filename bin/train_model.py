import molurus
from molurus import config_dict

from tkgl import models
from tkgl.trials import TkgrTrial


def main():
    parser = config_dict.ArgumentParser()
    parser.add_argument("--save-folder-path")
    args = parser.parse_args()
    cfg = config_dict.load(args.cfg, args.overrides)

    trial = TkgrTrial(args.save_folder_path)
    trial.validate_and_dump_config(cfg)

    datasets, vocabs = trial.load_datasets_and_vocab(cfg["data"])
    model = models.build_model(
        cfg["model"], num_ents=len(vocabs["ent"]), num_rels=len(vocabs["rel"])
    )
    criterion = molurus.smart_call(model.build_criterion, cfg["model"])

    state_dict = trial.train_model(
        model,
        datasets["train"],
        criterion,
        val_data=datasets["val"],
        tr_cfg=cfg["training"],
    )
    metric_dataframe = trial.eval_model(model, datasets["test"], state_dict)
    print(metric_dataframe.T * 100)


if __name__ == "__main__":
    main()
