import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionEncoder(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(VisionEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # We can add a pooling layer to reduce the spatial dimensions
        self.pool = nn.MaxPool2d(2, 2)
        
        self.gap = nn.AdaptiveAvgPool2d((7, 7))  # Global Average Pooling to get a fixed-size output
        
        # Update the input size of the fully connected layer to match the output shape of conv2
        # self.fc1 = nn.Linear(64 * 7 * 7, output_dim)  # 64 channels, 8x8 spatial dimensions after pooling
        self.fc1 = nn.Linear(128 * 7* 7, output_dim)  # 64 channels, 8x8 spatial dimensions after pooling

    def forward(self, x):
        # Convolutional layers to process image data
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        return x


