import copy
import itertools
import logging
import os
import re
from collections import defaultdict
from typing import Counter, Dict, Hashable, List, Optional, Tuple, TypedDict

import dgl
import torch
from dgl.udf import EdgeBatch
from tallow.data import datasets, vocabs

logger = logging.getLogger(__name__)


class Quadruple(TypedDict):
    subj: Hashable
    rel: Hashable
    obj: Hashable
    mmt: Hashable


class TkgRExample(TypedDict):
    hist_graphs: List[dgl.DGLGraph]
    subj: torch.Tensor
    rel: torch.Tensor
    obj: torch.Tensor
    quadruples: List[Quadruple]


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
    triplets: List[Quadruple], vocabs: Dict[str, vocabs.Vocab]
) -> dgl.DGLGraph:
    src_ids, dst_ids, rel_ids = [], [], []
    r2e = defaultdict(set)
    for trip in triplets:
        u = vocabs["ent"].to_id(trip["subj"])
        v = vocabs["ent"].to_id(trip["obj"])
        e = vocabs["rel"].to_id(trip["rel"])
        src_ids.append(u)
        dst_ids.append(v)
        rel_ids.append(e)
        r2e[e].update([u, v])

    node_ids = vocabs["ent"].to_id(vocabs["ent"])

    graph = dgl.graph([])

    rel_node_ids = []
    rel_len = []
    for rel_id in range(len(vocabs["rel"])):
        rel_node_ids.extend(r2e[rel_id])
        rel_len.append(len(r2e[rel_id]))
    graph.rel_node_ids = rel_node_ids
    graph.rel_len = rel_len

    graph.add_nodes(len(node_ids), data={"ent_id": torch.as_tensor(node_ids)})
    graph.add_edges(
        src_ids,
        dst_ids,
        data={
            "rel_id": torch.as_tensor(rel_ids),
        },
    )

    # external precompute value
    in_deg = graph.in_degrees()
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    degree_norm = torch.div(1.0, in_deg)
    graph.ndata["norm"] = degree_norm.view(-1, 1)

    def edge_norm(edges: EdgeBatch):
        return {"norm": edges.src["norm"] * edges.dst["norm"]}

    graph.apply_edges(edge_norm)

    return graph


QUAD_RE = re.compile(
    r"^(?P<subj>\d+)\s+(?P<rel>\d+)\s+(?P<obj>\d+)\s+(?P<time>\d+)"
)


def load_quadruples(
    quadruple_file: str, bidirectional: bool
) -> List[Quadruple]:
    quadruples: List[Quadruple] = []
    with open(quadruple_file) as fp:
        for line in fp:
            match = QUAD_RE.search(line)
            subj, rel, obj, time = match.group("subj", "rel", "obj", "time")
            quadruples.append(Quadruple(subj=subj, rel=rel, obj=obj, mmt=time))

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
    # There should not have a vocab for time field.
    # Times in val and test are unknown for training apparently.
    # sorted_times = sorted(set(field_values_dict["time"]), key=int)
    # vocabs["time"] = vocab.build_from_sequence(sorted_times)
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
        data_file = os.path.join(data_folder, f"{subset}.txt")
        quadruples[subset] = load_quadruples(data_file, bidirectional)
        num_edges = len(quadruples[subset])
        logger.info(f"# {subset} Edges {num_edges}")

    vocab_quads = copy.deepcopy(quadruples["train"])
    if different_unknowns:
        vocab_quads.extend(quadruples["val"])
        vocab_quads.extend(quadruples["test"])
    vocabs = build_vocab_from_quadruples(vocab_quads)

    temporal_quads = {}
    for subset in subsets:
        temporal_quads[subset] = groupby_temporal(quadruples[subset])

    datasets = {}
    if complement_val_and_test:
        datasets["train"] = ExtrapolateTKGDataset(
            temporal_quads["train"], hist_len, vocabs
        )
        datasets["val"] = ExtrapolateTKGDataset(
            temporal_quads["val"],
            hist_len,
            vocabs,
            temporal_quads["train"][-hist_len:],
        )
        datasets["test"] = ExtrapolateTKGDataset(
            temporal_quads["test"],
            hist_len,
            vocabs,
            temporal_quads["val"][-hist_len:],
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
