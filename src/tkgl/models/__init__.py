import logging
from typing import Dict, Type

import molurus

from tkgl.models.rerank import RerankTkgrModel
from tkgl.models.tkgr_model import TkgrModel

__all__ = ["TkgrModel"]

logger = logging.getLogger(__name__)


def build_model(cfg: Dict, **kwargs) -> TkgrModel:
    model_arch = cfg.pop("arch")
    model_class: Type[TkgrModel] = molurus.import_get(model_arch)

    if issubclass(model_class, RerankTkgrModel):
        backbone_cfg = cfg.pop("backbone")
        for k, v in cfg.items():
            dict.setdefault(backbone_cfg, k, v)

        cfg["backbone"] = build_model(backbone_cfg, **kwargs)

    logger.info(f"Model: {model_class.__name__}")
    model = molurus.smart_call(model_class, cfg, **kwargs)
    return model
