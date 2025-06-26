import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EmotionEfficientNet(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionEfficientNet, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        # Replace final layer with output layer for 7 classes
        in_features = self.base_model._fc.in_features
        self.base_model._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
