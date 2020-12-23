import torch
from torch import nn
import torchvision.models as models

class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class Backbone(nn.Module):
    def __init__(self, architecture, num_classes):
        super(Backbone, self).__init__()
        network = getattr(models, architecture)(pretrained=True)
        in_features = network.classifier._modules['6'].in_features
        network.classifier._modules['6'] = nn.Linear(in_features, num_classes)
        self.network = network
    
    def forward(self, x):
        return self.network(x)

def build_backbone(cfg, num_classes):
    architecture = cfg.MODEL.BACKBONE.ARCHITECTURE
    num_classes = cfg.MODEL.NUM_CLASSES
    backbone = Backbone(architecture, num_classes)
    return backbone
