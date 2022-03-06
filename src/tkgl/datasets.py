import copy
import itertools
import logging
import os
import re
from typing import Counter, Dict, List, Optional, Tuple

import dgl
import torch
from tallow.data import datasets, vocabs

from tkgl.data import Quadruple, TkgRExample

logger = logging.getLogger(__name__)


class ExtrapolateTKGDataset(datasets.Dataset[TkgRExample]):
    def __init__(
        self,
        temporal_quads: List[List[Quadruple]],
        hist_len: Optional[int],
        vocabs: Dict[str, vocabs.Vocab],
        prepend_quads: List[List[Quadruple]] = None,
    ) -> None:
        super().__init__()
        self._temporal_quads = temporal_quads
        self._hist_len = hist_len
        self._vocabs = vocabs

        if prepend_quads is None:
            prepend_quads = []
        else:
            assert len(prepend_quads) == hist_len
        self._prepend_quads = prepend_quads
        self._hist_graphs = [
            build_knowledge_graph(quads, self._vocabs)
            for quads in self._temporal_quads
        ]

    def __iter__(self):
        total_quads = self._prepend_quads + self._temporal_quads
        for future in range(
            len(self._prepend_quads) + 1, len(self._temporal_quads)
        ):
            xlice = slice(max(future - self._hist_len, 0), future)
            hist_graphs = self._hist_graphs[xlice]
            quadruples = total_quads[future]
            subjs = [quad["subj"] for quad in quadruples]
            rels = [quad["rel"] for quad in quadruples]
            objs = [quad["obj"] for quad in quadruples]

            example = TkgRExample(
                hist_graphs=hist_graphs,
                subj=torch.from_numpy(self._vocabs["ent"](subjs)),
                rel=torch.from_numpy(self._vocabs["rel"](rels)),
                obj=torch.from_numpy(self._vocabs["ent"](objs)),
                quadruples=quadruples,
            )
            yield datasets.Batch(example)

    def __len__(self):
        if self._prepend_quads:
            return len(self._temporal_quads)
        else:
            return len(self._temporal_quads) - 1


def groupby_temporal(quadruples: List[Quadruple]) -> List[List[Quadruple]]:
    mmt_getter = lambda quad: int(quad["mmt"])
    consecutive_quads = sorted(quadruples, key=mmt_getter)
    ret = []
    for _, quads in itertools.groupby(consecutive_quads, key=mmt_getter):
        quads = list(quads)
        if quads:
            ret.append(quads)
    return ret


def build_knowledge_graph(
    quadruples: List[Quadruple],
    vocabs: Dict[str, vocabs.Vocab],
) -> dgl.DGLGraph:
    src_ids, dst_ids, rel_ids = [], [], []
    for quad in quadruples:
        u = vocabs["ent"].to_id(quad["subj"])
        v = vocabs["ent"].to_id(quad["obj"])
        e = vocabs["rel"].to_id(quad["rel"])
        src_ids.append(u)
        dst_ids.append(v)
        rel_ids.append(e)

    node_ids = vocabs["ent"].to_id(vocabs["ent"])

    graph = dgl.graph((src_ids, dst_ids), num_nodes=len(node_ids))
    graph.ndata["eid"] = torch.tensor(node_ids)
    graph.edata["rid"] = torch.tensor(rel_ids)

    return graph


def load_quadruples(
    quadruple_file: str, bidirectional: bool
) -> List[Quadruple]:
    quadruples: List[Quadruple] = []
    with open(quadruple_file) as fp:
        for line in fp:
            line_splits = line.strip().split("\t")
            subj, rel, obj, mmt, *_ = line_splits
            quadruples.append(Quadruple(subj=subj, rel=rel, obj=obj, mmt=mmt))

    if bidirectional:
        reversed_quads = []
        for quad in quadruples:
            rev_quad = Quadruple(
                subj=quad["obj"],
                rel="REVERSE # " + quad["rel"],
                obj=quad["subj"],
                mmt=quad["mmt"],
            )
            reversed_quads.append(rev_quad)
        quadruples += reversed_quads

    return quadruples


def build_vocab_from_quadruples(
    quadruples: List[Quadruple],
) -> Dict[str, vocabs.Vocab]:
    ents = Counter()
    rels = Counter()
    for quad in quadruples:
        ents.update([quad["subj"], quad["obj"]])
        rels.update([quad["rel"]])

    presv_toks = {"unk": "[UNK]"}
    tknzs = {
        "ent": vocabs.counter_vocab(ents, presv_toks),
        "rel": vocabs.counter_vocab(rels, presv_toks),
    }
    return tknzs


def load_tkg_dataset(
    data_folder: str,
    hist_len: int,
    bidirectional: bool,
    different_unknowns: bool,
    complement_val_and_test: bool,
    shuffle: bool,
) -> Tuple[Dict[str, ExtrapolateTKGDataset], Dict[str, vocabs.Vocab]]:
    subsets = ("train", "val", "test")
    quadruples: Dict[str, List[Quadruple]] = {}
    for subset in subsets:
        data_file = os.path.join(data_folder, f"{subset}.tsv")
        quadruples[subset] = load_quadruples(data_file, bidirectional)
    logger.info(
        "# Edges: Train: %d  Val: %d  Test: %d",
        len(quadruples["train"]),
        len(quadruples["val"]),
        len(quadruples["test"]),
    )

    vocab_quads = copy.deepcopy(quadruples["train"])
    if different_unknowns:
        vocab_quads.extend(quadruples["val"])
        vocab_quads.extend(quadruples["test"])
    vocabs = build_vocab_from_quadruples(vocab_quads)

    temporal_quads = {}
    for subset in subsets:
        temporal_quads[subset] = groupby_temporal(quadruples[subset])

    logger.info(
        "# Timestamps: Train: %d  Val: %d  Test: %d",
        len(temporal_quads["train"]),
        len(temporal_quads["val"]),
        len(temporal_quads["test"]),
    )

    datasets = {}
    if complement_val_and_test:
        hist_quads = []
        datasets["train"] = ExtrapolateTKGDataset(
            temporal_quads["train"], hist_len, vocabs
        )
        hist_quads.extend(temporal_quads["train"])
        datasets["val"] = ExtrapolateTKGDataset(
            temporal_quads["val"], hist_len, vocabs, hist_quads[-hist_len:]
        )
        hist_quads.extend(temporal_quads["val"])
        datasets["test"] = ExtrapolateTKGDataset(
            temporal_quads["test"], hist_len, vocabs, hist_quads[-hist_len:]
        )
    else:
        for subset in subsets:
            dataset = ExtrapolateTKGDataset(
                temporal_quads[subset], hist_len, vocabs
            )
            datasets[subset] = dataset
    if shuffle:
        datasets["train"] = datasets["train"].shuffle()
    return datasets, vocabs
