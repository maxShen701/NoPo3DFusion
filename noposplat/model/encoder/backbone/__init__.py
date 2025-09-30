from typing import Any
import torch.nn as nn

from noposplat.model.encoder.backbone.backbone import Backbone
from noposplat.model.encoder.backbone.backbone_croco_multiview import AsymmetricCroCoMulti
from noposplat.model.encoder.backbone.backbone_dino import BackboneDino, BackboneDinoCfg
from noposplat.model.encoder.backbone.backbone_resnet import BackboneResnet, BackboneResnetCfg
from noposplat.model.encoder.backbone.backbone_croco import AsymmetricCroCo, BackboneCrocoCfg

BACKBONES: dict[str, Backbone[Any]] = {
    "resnet": BackboneResnet,
    "dino": BackboneDino,
    "croco": AsymmetricCroCo,
    "croco_multi": AsymmetricCroCoMulti,
}

BackboneCfg = BackboneResnetCfg | BackboneDinoCfg | BackboneCrocoCfg


def get_backbone(cfg: BackboneCfg, d_in: int = 3) -> nn.Module:
    return BACKBONES[cfg.name](cfg, d_in)
