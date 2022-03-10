import argparse
import logging
import os
from typing import List, Set, Tuple

import molurus
import torch
from molurus import config_dict
from tallow.nn import forwards

from tkgl.datasets import load_tkg_dataset
from tkgl.models import tkgr_model
from tkgl.trials import TkgrTrial

logger = logging.getLogger(__name__)


def load_cfg(checkpoint_path: str, config_path: str = None):
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(checkpoint_path), "config.yml"
        )

    cfg = config_dict.load(open(config_path))
    return cfg


def load_model(model_cfg, checkpoint_path: str, **kwargs):
    model = tkgr_model.build_model(model_cfg, **kwargs)
    model.load_state_dict(torch.load(checkpoint_path)["model"])
    return model


def chains_to_group(tuples: List[Tuple[int, int]]) -> List[Set[int]]:
    increasing_sets: List[Set[int]] = []
    while True:
        for tup in tuples:
            for inc_set in increasing_sets:
                if tup[0] in inc_set:
                    inc_set.add(tup[0])
                if tup[1] in inc_set:
                    inc_set.add(tup[1])

        total_sets: Set[List] = set()
        for inc_set in increasing_sets:
            total_sets.add(sorted(inc_set))
        new_increasing_sets = [set(s) for s in total_sets]
        if sum(map(len, new_increasing_sets)) == sum(map, len(increasing_sets)):
            return new_increasing_sets
        increasing_sets = new_increasing_sets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("regcn")
    parser.add_argument("rerank")
    args = parser.parse_args()

    regcn_cfg = load_cfg(args.regcn)
    rerank_cfg = load_cfg(args.rerank)
    assert regcn_cfg["data"] == rerank_cfg["data"]
    data_cfg = rerank_cfg["data"]

    datasets, vocabs = molurus.smart_call(
        load_tkg_dataset, data_cfg, shuffle=False
    )

    regcn = load_model(
        regcn_cfg["model"],
        args.regcn,
        num_ents=len(vocabs["ent"]),
        num_rels=len(vocabs["rel"]),
    )
    rerank = load_model(
        rerank_cfg["model"],
        args.rerank,
        num_ents=len(vocabs["ent"]),
        num_rels=len(vocabs["rel"]),
    )

    def triplet_string(s: torch.Tensor, r: torch.Tensor, o: torch.Tensor):
        s = vocabs["ent"].to_token(s.item())
        r = vocabs["rel"].to_token(r.item())
        o = vocabs["ent"].to_token(o.item())
        return "({},\t{},\t{})".format(s, r, o)

    def print_ent_trips(ent):
        ent_mask = gold_trips[:, 0] == ent
        ent_trips = gold_trips[ent_mask]
        for s, r, o in ent_trips:
            print(triplet_string(s, r, o))

    for model_inputs in datasets["test"]:
        with torch.set_grad_enabled(False):
            regcn_outputs = forwards.module_forward(regcn, model_inputs)
            rerank_outputs = forwards.module_forward(rerank, model_inputs)
        regcn_opred = torch.argmax(regcn_outputs["obj_logit"], dim=-1)
        rerank_opred = torch.argmax(rerank_outputs["obj_logit"], dim=-1)

        diff_mask = regcn_opred != rerank_opred
        num_diffs = torch.count_nonzero(diff_mask).item()
        logger.info("Find %d differences", num_diffs)

        regcn_false_mask = regcn_opred != model_inputs["obj"]
        rerank_true_mask = rerank_opred == model_inputs["obj"]
        target_mask = torch.bitwise_and(regcn_false_mask, rerank_true_mask)
        num_targets = torch.count_nonzero(target_mask).item()
        logger.info("Find %d target_samples", num_targets)
        target_indexes = torch.nonzero(target_mask)[:, -1]
        for target in target_indexes:
            gold_trips = torch.stack(
                [model_inputs[f] for f in ["subj", "rel", "obj"]], dim=1
            )

            subj = model_inputs["subj"][target]
            rel = model_inputs["rel"][target]
            print("REGCN & RERANK ")
            print(triplet_string(subj, rel, regcn_opred[target]))
            print(triplet_string(subj, rel, rerank_opred[target]))

            print("Subj Context: ")
            print_ent_trips(subj)

            print("False Context: ")
            print_ent_trips(regcn_opred[target])

            print("Truth Context: ")
            print_ent_trips(rerank_opred[target])

            print()
            print("=" * 20)
            input("Enter to continue")


if __name__ == "__main__":
    main()
