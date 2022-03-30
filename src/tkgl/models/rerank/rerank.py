import copy
import logging
import os
from typing import List

import dgl
import molurus
import torch
from molurus import hierdict
from torch.nn import functional as f

from tkgl.models.tkgr_model import TkgrModel

logger = logging.getLogger(__name__)


class RerankTkgrModel(torch.nn.Module):
    def __init__(
        self,
        num_ents: int,
        num_rels: int,
        pretrained_backbone: str,
        finetune: bool,
        config_path: str = None,
    ):
        super().__init__()
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(pretrained_backbone), "config.yml"
            )
        bb_cfg = hierdict.load(open(config_path))
        self.backbone: TkgrModel = molurus.smart_instantiate(
            bb_cfg["model"], num_ents=num_ents, num_rels=num_rels
        )

        self.finetune = finetune
        if pretrained_backbone:
            logger.info("Load backbone from %s", pretrained_backbone)
            backbone_state = torch.load(pretrained_backbone)["model"]
            self.backbone.load_state_dict(backbone_state)
        self.backbone.requires_grad_(self.finetune)

    @property
    def hidden_size(self):
        return self.backbone.hidden_size


class RerankLoss(torch.nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self._alpha = alpha

    def forward(
        self,
        obj_logit: torch.Tensor,
        obj_logit_orig: torch.Tensor,
        obj: torch.Tensor,
    ):
        obj_orig_loss = f.cross_entropy(obj_logit_orig, obj)
        obj_loss = f.cross_entropy(obj_logit, obj)
        return self._alpha * obj_orig_loss + (1 - self._alpha) * obj_loss
