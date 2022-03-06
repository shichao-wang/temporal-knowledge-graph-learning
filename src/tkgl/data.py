from typing import Hashable, List, TypedDict

import dgl
import torch


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
