import os
from typing import Callable, List, Mapping

import optuna
from molurus import hierdict
from train_model import train

OptunaTuneFunction = Callable[[str, hierdict.HierDict], float]


class OptunaTuner:
    def __init__(
        self,
        save_folder_path: str,
        train_model: OptunaTuneFunction,
        base_cfg: hierdict.HierDict,
        hpspace: Mapping,
        studyname: str = "optuna",
        direction: str = "maximize",
    ) -> None:
        os.makedirs(save_folder_path, exist_ok=True)
        self.save_folder_path = save_folder_path
        self.train_model = train_model
        self.base_cfg = base_cfg
        self.hpspace = hpspace

        storage = f"sqlite:///{self.save_folder_path}/{studyname}.db"
        self.study = optuna.create_study(
            storage=storage,
            study_name=studyname,
            load_if_exists=True,
            direction=direction,
        )

    def execute(
        self, num_trials: int = None, num_seconds: int = None
    ) -> optuna.Trial:
        try:
            self.study.optimize(self._objective, num_trials, num_seconds)
        except KeyboardInterrupt:
            pass
        best_trial = self.study.best_trial
        return best_trial

    def _objective(self, trial: optuna.Trial):
        trial_folder_path = os.path.join(
            self.save_folder_path, str(trial.number)
        )
        trial_cfg = suggest_cfg(self.base_cfg, trial, self.hpspace)
        return self.train_model(trial_folder_path, trial_cfg)


def suggest_cfg(
    base_cfg: hierdict.HierDict, trial: optuna.Trial, hp_space: Mapping
):
    def suggest(k: str):
        if isinstance(hp_space[k], List) and len(hp_space[k]) > 3:
            return trial.suggest_categorical(k, hp_space[k])
        if isinstance(base_cfg[k], str):
            return trial.suggest_categorical(k, hp_space[k])

        low, high, step = hp_space[k]
        if isinstance(base_cfg[k], float):
            return trial.suggest_float(k, low, high, step=step)
        if isinstance(base_cfg[k], int):
            return trial.suggest_int(k, low, high, step)
        raise ValueError()

    cfg = base_cfg.copy()
    for key in hp_space:
        cfg[key] = suggest(key)
    return cfg


def main():
    parser = hierdict.ArgumentParser()
    parser.add_argument("--save-folder-path")
    parser.add_argument("--num-trials", default=20)
    args = parser.parse_args()
    base_cfg = hierdict.parse_args(args.cfg, args.overrides)
    hpspace = base_cfg.pop("hpspace")

    tuner = OptunaTuner(args.save_folder_path, train, base_cfg, hpspace)
    best_trial = tuner.execute(args.num_trials)
    print(best_trial)


if __name__ == "__main__":
    main()
