import copy
import logging
from typing import List

import dgl
import torch
from torch.nn import functional as tfn

from tkgl.models.tkgr_model import TkgrModel

logger = logging.getLogger(__name__)


class RerankTkgrModel(torch.nn.Module):
    def __init__(
        self,
        backbone: TkgrModel,
        finetune: bool,
        pretrained_backbone: str = None,
    ):
        super().__init__()
        self.backbone = backbone
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
    def __init__(self, alpha: float, beta: float):
        super().__init__()
        self._alpha = alpha
        self._beta = beta

    def forward(
        self,
        obj_logit: torch.Tensor,
        obj_logit_orig: torch.Tensor,
        obj: torch.Tensor,
        rel_logit: torch.Tensor,
        rel: torch.Tensor,
    ):
        obj_orig_loss = tfn.cross_entropy(obj_logit_orig, obj)
        rel_loss = tfn.cross_entropy(rel_logit, rel)
        obj_loss = tfn.cross_entropy(obj_logit, obj)
        orig_loss = self._alpha * obj_orig_loss + (1 - self._alpha) * rel_loss
        return self._beta * orig_loss + (1 - self._beta) * obj_loss
