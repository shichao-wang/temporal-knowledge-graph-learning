import logging
from typing import Dict, Type

import molurus

from tkgl.models.rerank import RerankTkgrModel
from tkgl.models.tkgr_model import TkgrModel

__all__ = ["TkgrModel"]

logger = logging.getLogger(__name__)


def build_model(cfg: Dict, **kwargs) -> TkgrModel:
    model_cfg = cfg.copy()
    model_arch = model_cfg.pop("arch")
    model_class: Type[TkgrModel] = molurus.import_get(model_arch)

    if issubclass(model_class, RerankTkgrModel):
        backbone_cfg = model_cfg.pop("backbone")
        for k, v in model_cfg.items():
            dict.setdefault(backbone_cfg, k, v)

        model_cfg["backbone"] = build_model(backbone_cfg, **kwargs)

    logger.info(f"Model: {model_class.__name__}")
    model = molurus.smart_call(model_class, model_cfg, **kwargs)
    return model
