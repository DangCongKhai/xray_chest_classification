import torch
import torch.nn as nn
from torchvision import models


class SimpleEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.DEFAULT 
        self.model = models.efficientnet_b0(weights=weights)
        # Unfreeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # replace classifer
        num_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class EfficientAdvanced(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.model = models.efficientnet_b0(weights=weights)
        # Unfreeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # replace classifer
        num_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2), 
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3,),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)
