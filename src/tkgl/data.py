from typing import Dict, Hashable, List, TypedDict

import dgl
import torch


class Quadruple(TypedDict):
    subj: Hashable
    rel: Hashable
    obj: Hashable
    mmt: Hashable


class TkgrFeature(TypedDict):
    hist_graphs: List[dgl.DGLGraph]
    subj: torch.Tensor
    rel: torch.Tensor
    obj: torch.Tensor
    sr_dict: Dict[int, Dict[int, List[int]]]
    so_dict: Dict[int, Dict[int, List[int]]]
    # all_obj_mask: torch.Tensor
    # all_rel_mask: torch.Tensor
