import torch
from torch import nn

from .backbone import build_backbone
from .loss import build_loss

class Model(nn.Module):
    def __init__(self, cfg, is_train):
        super(Model, self).__init__()
        self.backbone = build_backbone(cfg, cfg.MODEL.NUM_CLASSES)
        self.loss = build_loss(cfg)
    
    def forward(self, images, targets=None):
        logits = self.backbone(images)
        if self.training:
            return self.loss(logits, targets)
        else:
            return logits
