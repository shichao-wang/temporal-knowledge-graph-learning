import dataclasses
import itertools
import operator
import os
import re
from collections import defaultdict
from typing import Counter, Dict, Hashable, List, Tuple

import dgl
import py_helpers
import torch
from dgl.udf import EdgeBatch
from torch_helpers import Vocab, vocab
from torch_helpers.datasets import Batch, Dataset


@dataclasses.dataclass()
class Quadruple:
    head: Hashable
    rel: Hashable
    tail: Hashable
    time: Hashable


@dataclasses.dataclass()
class QuadrupleBatchInput(Batch):
    snapshots: List[dgl.DGLGraph]
    quadruples: List[Quadruple]
    head: torch.Tensor
    rel: torch.Tensor
    tail: torch.Tensor
    future_id: int

    def to(self, device: str):
        self.snapshots = [snap.to(device) for snap in self.snapshots]
        super().to(device)
        return self


class QuadrupleLoader:
    def __init__(
        self, vocabs: Dict[str, Vocab], hist_len: int, bidirectional: bool
    ) -> None:
        self._vocabs = vocabs
        self._hist_len = hist_len
        self._bidirectional = bidirectional

    def load(self, quadruple_file: str) -> Dataset:
        quadruples = load_quadruples(quadruple_file, self._bidirectional)
        temporal_quads = self.groupby_temporal(quadruples)
        timestamps = sorted(temporal_quads.keys(), key=int)
        snapshots = [
            self.build_graph(temporal_quads[time]) for time in timestamps
        ]

        def batch(future: int):
            hist_snaps = snapshots[max(future - self._hist_len, 0) : future]
            quadruples = temporal_quads[timestamps[future]]

            heads = [quad.head for quad in quadruples]
            rels = [quad.rel for quad in quadruples]
            tails = [quad.tail for quad in quadruples]

            return QuadrupleBatchInput(
                snapshots=hist_snaps,
                quadruples=quadruples,
                head=torch.as_tensor(
                    self._vocabs["entity"].convert_tokens_to_ids(heads)
                ),
                rel=torch.as_tensor(
                    self._vocabs["relation"].convert_tokens_to_ids(rels)
                ),
                tail=torch.as_tensor(
                    self._vocabs["entity"].convert_tokens_to_ids(tails)
                ),
                future_id=future,
            )

        dataset = Dataset([batch(i) for i in range(1, len(snapshots))])
        return dataset

    def groupby_temporal(
        self, quadruples: List[Quadruple]
    ) -> Dict[str, List[Quadruple]]:
        time_getter = operator.attrgetter("time")
        ordered_quads = sorted(quadruples, key=time_getter)
        ret = {}
        for time, quads in itertools.groupby(ordered_quads, key=time_getter):
            ret[time] = list(quads)
        return ret

    def build_graph(self, triplets: List[Quadruple]) -> dgl.DGLGraph:
        src_ids, dst_ids, rel_ids = [], [], []
        r2e = defaultdict(set)
        for trip in triplets:
            u = self._vocabs["entity"].convert_token_to_id(trip.head)
            v = self._vocabs["entity"].convert_token_to_id(trip.tail)
            e = self._vocabs["relation"].convert_token_to_id(trip.rel)
            src_ids.append(u)
            dst_ids.append(v)
            rel_ids.append(e)
            r2e[e].update([u, v])

        node_ids = self._vocabs["entity"].convert_tokens_to_ids(
            self._vocabs["entity"]
        )

        graph = dgl.graph([])

        rel_node_ids = []
        rel_len = []
        for rel_id in range(self._vocabs["relation"].max_index):
            rel_node_ids.extend(r2e[rel_id])
            rel_len.append(len(r2e[rel_id]))
        graph.rel_node_ids = rel_node_ids
        graph.rel_len = rel_len

        graph.add_nodes(
            len(node_ids),
            data={"ent_id": torch.as_tensor(node_ids)},
        )
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


def build_vocabs(quadruple_file: str, bidirectional: str) -> Dict[str, Vocab]:
    quadruples = load_quadruples(quadruple_file, bidirectional)
    return build_vocab_from_quadruples(quadruples)


QUAD_RE = re.compile(
    r"^(?P<head>\d+)\s+(?P<rel>\d+)\s+(?P<tail>\d+)\s+(?P<time>\d+)"
)


def load_quadruples(
    quadruple_file: str, bidirectional: bool
) -> Dict[str, Vocab]:
    quadruples: List[Quadruple] = []
    with py_helpers.auto_open(quadruple_file) as fp:
        for line in fp:
            match = QUAD_RE.search(line)
            head, tail, rel, time = match.group("head", "tail", "rel", "time")
            quadruples.append(Quadruple(head, rel, tail, time))

    if bidirectional:
        reversed_quads = []
        for quad in quadruples:
            reversed_quads.append(
                Quadruple(
                    head=quad.tail,
                    rel="reverse" + quad.rel,
                    tail=quad.head,
                    time=quad.time,
                )
            )
        quadruples = quadruples + reversed_quads

    return quadruples


def build_vocab_from_quadruples(
    quadruples: List[Quadruple],
) -> Dict[str, Vocab]:
    field_values_dict = {
        field.name: [getattr(quad, field.name) for quad in quadruples]
        for field in dataclasses.fields(Quadruple)
    }

    def counter_vocab(counter: Counter):
        return vocab.build_from_counter(
            counter,
            special_tokens={"[UNK]": 1, "[PAD]": 0},
            identified_tokens={"pad": "[PAD]", "unk": "[UNK]"},
        )

    counters = {
        "entity": Counter(
            field_values_dict["head"] + field_values_dict["tail"]
        ),
        "relation": Counter(field_values_dict["rel"]),
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
) -> Tuple[Dict[str, Dataset], Dict[str, Vocab]]:
    vocabs = build_vocabs(os.path.join(data_folder, "train.txt"), bidirectional)
    loader = QuadrupleLoader(vocabs, hist_len, bidirectional)
    datasets = {
        s: loader.load(os.path.join(data_folder, f"{s}.txt"))
        for s in ["train", "val", "test"]
    }
    return datasets, vocabs
