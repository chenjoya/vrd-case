import torch
from torch import nn
import torchvision.models as models

class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class Backbone(nn.Module):
    def __init__(self, architecture, num_classes, prior):
        super(Backbone, self).__init__()
        network = getattr(models, architecture)(pretrained=True)
        in_features = network.classifier._modules['6'].in_features
        network.classifier._modules['6'] = nn.Linear(in_features, num_classes)
        bias = -torch.log((1 - prior) / prior)
        network.classifier._modules['6'].bias.data = bias
        self.network = network
    
    def forward(self, x):
        return self.network(x)

def build_backbone(architecture, num_classes, prior):
    return Backbone(architecture, num_classes, prior)
