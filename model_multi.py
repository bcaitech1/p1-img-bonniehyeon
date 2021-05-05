import torch.nn as nn
from timm.models.layers.classifier import ClassifierHead
import timm


class MyModel_seperation_next(nn.Module):
    def __init__(self,outdim):
        super().__init__()
        self.model = timm.create_model('resnext50_32x4d', pretrained=True)
        
        num_fits = self.model.num_features
        self.layer = ClassifierHead(num_fits,3)             

    def forward(self, x):
        x = self.model.forward_features(x)       
        out = self.layer(x)
        return out

