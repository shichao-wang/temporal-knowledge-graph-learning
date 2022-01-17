import dataclasses
import itertools
import operator
import os
import re
from collections import defaultdict
from typing import Counter, Dict, Hashable, Iterable, List, Tuple, TypedDict

import dgl
import py_helpers
import torch
import torch_helpers as th
from dgl.udf import EdgeBatch


@dataclasses.dataclass()
class Quadruple:
    subj: Hashable
    rel: Hashable
    obj: Hashable
    time: Hashable


class QuadrupleBatchInput(TypedDict):
    snapshots: List[dgl.DGLGraph]
    quadruples: List[Quadruple]
    time: torch.Tensor
    subj: torch.Tensor
    rel: torch.Tensor
    obj: torch.Tensor
    future_id: int


def groupby_temporal(quadruples: List[Quadruple]) -> Dict[str, List[Quadruple]]:
    time_getter = operator.attrgetter("time")
    ordered_quads = sorted(quadruples, key=time_getter)
    ret = {}
    for time, quads in itertools.groupby(ordered_quads, key=time_getter):
        ret[time] = list(quads)
    return ret


def build_knowledge_graph(
    triplets: List[Quadruple], vocabs: Dict[str, th.tokenizers.LookupTokenizer]
) -> dgl.DGLGraph:
    src_ids, dst_ids, rel_ids = [], [], []
    r2e = defaultdict(set)
    for trip in triplets:
        u = vocabs["ent"].to_id(trip.subj)
        v = vocabs["ent"].to_id(trip.obj)
        e = vocabs["rel"].to_id(trip.rel)
        src_ids.append(u)
        dst_ids.append(v)
        rel_ids.append(e)
        r2e[e].update([u, v])

    node_ids = vocabs["ent"].to_id(vocabs["ent"].get_vocab())

    graph = dgl.graph([])

    rel_node_ids = []
    rel_len = []
    for rel_id in range(vocabs["rel"].vocab_size):
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

    in_deg = graph.in_degrees()
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    degree_norm = torch.div(1.0, in_deg)
    graph.ndata["norm"] = degree_norm.view(-1, 1)

    def edge_norm(edges: EdgeBatch):
        return {"norm": edges.src["norm"] * edges.dst["norm"]}

    graph.apply_edges(edge_norm)

    return graph


class ExtrapolateTKGDataset(th.datasets.IterableDataset[QuadrupleBatchInput]):
    def __init__(
        self,
        quadruples: Iterable[Quadruple],
        vocabs: Dict[str, th.tokenizers.LookupTokenizer],
        hist_len: int,
    ) -> None:
        super().__init__()
        self._temporal_quads = groupby_temporal(quadruples)
        self._vocabs = vocabs
        self._hist_len = hist_len
        self._timestamps = sorted(self._temporal_quads, key=int)
        self._snapshots = [
            build_knowledge_graph(self._temporal_quads[time], self._vocabs)
            for time in self._timestamps
        ]

    def __iter__(self):
        for future in range(1, len(self._snapshots)):
            hist_snaps = self._snapshots[
                max(future - self._hist_len, 0) : future
            ]
            quadruples = self._temporal_quads[self._timestamps[future]]
            subjs = [quad.subj for quad in quadruples]
            rels = [quad.rel for quad in quadruples]
            objs = [quad.obj for quad in quadruples]
            times = [quad.time for quad in quadruples]

            batch = QuadrupleBatchInput(
                snapshots=hist_snaps,
                quadruples=quadruples,
                subj=self._vocabs["ent"](subjs),
                rel=self._vocabs["rel"](rels),
                obj=self._vocabs["ent"](objs),
                time=times,
                future_id=future,
            )
            yield th.datasets.Batch(batch)

    def __len__(self):
        return len(self._snapshots) - 1


def build_vocabs(
    quadruple_file: str, bidirectional: str
) -> Dict[str, th.tokenizers.LookupTokenizer]:
    quadruples = load_quadruples(quadruple_file, bidirectional)
    return build_vocab_from_quadruples(quadruples)


QUAD_RE = re.compile(
    r"^(?P<subj>\d+)\s+(?P<rel>\d+)\s+(?P<obj>\d+)\s+(?P<time>\d+)"
)


def load_quadruples(
    quadruple_file: str, bidirectional: bool
) -> List[Quadruple]:
    quadruples: List[Quadruple] = []
    with py_helpers.auto_open(quadruple_file) as fp:
        for line in fp:
            match = QUAD_RE.search(line)
            subj, rel, obj, time = match.group("subj", "rel", "obj", "time")
            quadruples.append(Quadruple(subj, rel, obj, time))

    if bidirectional:
        reversed_quads = []
        for quad in quadruples:
            reversed_quads.append(
                Quadruple(
                    subj=quad.obj,
                    rel="REVERSE # " + quad.rel,
                    obj=quad.subj,
                    time=quad.time,
                )
            )
        quadruples = quadruples + reversed_quads

    return quadruples


def build_vocab_from_quadruples(
    quadruples: List[Quadruple],
) -> Dict[str, th.tokenizers.LookupTokenizer]:
    field_values_dict = {
        field.name: [getattr(quad, field.name) for quad in quadruples]
        for field in dataclasses.fields(Quadruple)
    }

    def counter_vocab(counter: Counter):
        return th.tokenizers.CounterLookupTokenizer(
            counter, pad_tok="[PAD]", unk_tok="[UNK]"
        )

    counters = {
        "ent": Counter(field_values_dict["subj"] + field_values_dict["obj"]),
        "rel": Counter(field_values_dict["rel"]),
        "time": Counter(field_values_dict["time"]),
    }
    vocabs = {key: counter_vocab(ctr) for key, ctr in counters.items()}
    # There should not have a vocab for time field.
    # Times in val and test are unknown for training apparently.
    # sorted_times = sorted(set(field_values_dict["time"]), key=int)
    # vocabs["time"] = vocab.build_from_sequence(sorted_times)
    return vocabs


def load_tkg_dataset(
    data_folder: str,
    hist_len: int,
    bidirectional: bool,
) -> Tuple[
    Dict[str, ExtrapolateTKGDataset], Dict[str, th.tokenizers.LookupTokenizer]
]:
    vocabs = build_vocabs(os.path.join(data_folder, "train.txt"), bidirectional)
    datasets = {}
    for subset in ["train", "val", "test"]:
        data_file = os.path.join(data_folder, f"{subset}.txt")
        quadruples = load_quadruples(data_file, bidirectional)
        dataset = ExtrapolateTKGDataset(quadruples, vocabs, hist_len)
        datasets[subset] = dataset
    return datasets, vocabs
