import torch
from torch import nn
import torchvision.models as models

class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class Encoder(nn.Module):
    def __init__(self, architecture, num_channels):
        super(Encoder, self).__init__()
        network = getattr(models, architecture)(pretrained=True)
        in_features = network.fc.in_features
        network.fc = Identity()
        self.bnet = nn.Sequential(
            nn.Linear(in_features, num_channels),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels, num_channels),
            nn.ReLU(inplace=True),
        )
        self.unet = nn.Sequential(
            nn.Linear(in_features, num_channels),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels, num_channels),
            nn.ReLU(inplace=True),
        )
        self.network = network
        
    def forward(self, bimages, uimages):
        x = torch.cat([bimages, uimages])
        x = self.network(x).relu_()
        return self.bnet(x[:len(bimages)]), self.unet(x[len(bimages):])

def build_encoder(architecture, num_channels):
    return Encoder(architecture, num_channels)
