"""
cnn_detector.py
Convolutional Neural Network (CNN) for Vehicle Density Estimation & Visual Feature Extraction.
Processes road-camera footage frames, predicts vehicle counts and congestion levels,
and outputs visual spatial feature embeddings for downstream temporal models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrafficCNN(nn.Module):
    """
    Custom Deep Convolutional Network for Traffic Road Camera Analysis.
    Architecture:
      - 4 Convolutional blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool)
      - Adaptive Average Pooling
      - 64-dimensional Bottleneck Latent Feature Embedding
      - Dual Heads:
          1. Count / Density Regressor (continuous vehicle count)
          2. Congestion Classifier (4 classes: Free, Moderate, Heavy, Severe)
    """
    def __init__(self, in_channels=3, feature_dim=64, num_classes=4):
        super(TrafficCNN, self).__init__()
        
        # Block 1: 224x224 -> 112x112
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: 112x112 -> 56x56
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3: 56x56 -> 28x28
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 4: 28x28 -> 14x14
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global aggregation
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Latent spatial embedding layer
        self.fc_latent = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        
        # Head 1: Vehicle Count Regressor
        self.count_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Head 2: Traffic Congestion Level Classifier
        self.classifier_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def extract_features(self, x):
        """Extracts 64-dimensional visual feature embedding from video frame."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        latent = self.fc_latent(x)
        return latent

    def forward(self, x):
        latent = self.extract_features(x)
        pred_count = self.count_head(latent)
        pred_logits = self.classifier_head(latent)
        return pred_count, pred_logits, latent


if __name__ == "__main__":
    model = TrafficCNN()
    dummy_input = torch.randn(4, 3, 224, 224)
    pred_count, pred_logits, latent = model(dummy_input)
    print("CNN Architecture successfully verified:")
    print("  Input Shape:       ", dummy_input.shape)
    print("  Pred Count Shape:  ", pred_count.shape)
    print("  Pred Logits Shape: ", pred_logits.shape)
    print("  Latent Dim Shape:  ", latent.shape)
