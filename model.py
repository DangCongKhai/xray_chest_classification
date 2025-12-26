import torch
import torch.nn as nn
from torchvision import models

class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super(BaselineCNN, self).__init__()
        
        # Convolutional Block
        self.conv = nn.Conv2d(
            in_channels=3, out_channels=32,
            kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2)
        
        # Adaptive Pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Flatten & Classifier
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(
            in_features=32 * 4 * 4, 
            out_features=num_classes)
        
    def forward(self, x):
        # 1. Feature Extraction
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # 2. Downsampling
        x = self.adaptive_pool(x)
        
        # 3. Classification
        x = self.flatten(x)
        x = self.fc(x)
        return x

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
