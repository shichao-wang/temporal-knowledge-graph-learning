import argparse
import logging
import operator
import os
from functools import reduce

import tkgl

logging.getLogger(__name__)


def count_file(dfile: str):
    assert os.path.exists(dfile)

    quads = tkgl.datasets.load_quadruples(dfile, bidirectional=False)
    temporal_quads = tkgl.datasets.groupby_temporal(quads)

    obj_sets = {"subj": set(), "rel": set(), "obj": set(), "mmt": set()}
    for quad in quads:
        for field in ["subj", "rel", "obj", "mmt"]:
            obj_sets[field].add(quad[field])
    ent_set = obj_sets["subj"] | obj_sets["obj"]
    return {
        "quads": quads,
        "ent": ent_set,
        **obj_sets,
    }


subsets = ("train", "val", "test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder")
    args = parser.parse_args()

    data_folder = args.data_folder
    assert os.path.exists(data_folder)
    results = {}
    for subset in subsets:
        data_file = subset + ".txt"
        data_path = os.path.join(data_folder, data_file)
        subset_results = count_file(data_path)
        print(f"Count {subset}")
        num_edges = len(subset_results["quads"])
        print(f"#Edges: {num_edges}")
        num_entities = len(subset_results["ent"])
        print(f"#Entities: {num_entities}")
        num_relations = len(subset_results["rel"])
        print(f"#Relations: {num_relations}")
        num_histories = len(subset_results["mmt"])
        print(f"#History: {num_histories}")
        results[subset] = subset_results

    print("Total: ")
    num_edges = sum(len(results[s]["quads"]) for s in subsets)
    print(f"#Edges: {num_edges}")
    ent_set = reduce(operator.or_, (results[s]["ent"] for s in subsets), set())
    num_entities = len(ent_set)
    print(f"#Entities: {num_entities}")
    rel_set = reduce(operator.or_, (results[s]["rel"] for s in subsets), set())
    num_relations = len(rel_set)
    print(f"#Relations: {num_relations}")


if __name__ == "__main__":
    main()
