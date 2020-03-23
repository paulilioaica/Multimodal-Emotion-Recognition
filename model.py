import torch.nn as nn
import torch.nn.functional as F
import torch


class Video(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(kernel_size=3, padding=1, in_channels=3, out_channels=16),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=16),
                                     nn.MaxPool2d(kernel_size=2),
                                     nn.Conv2d(kernel_size=3, padding=1, in_channels=16, out_channels=32),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=32),
                                     nn.MaxPool2d(kernel_size=2),
                                     nn.Conv2d(kernel_size=3, padding=1, in_channels=32, out_channels=64),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=64),
                                     nn.MaxPool2d(kernel_size=2))
        self.linear = nn.Linear(in_features=43200, out_features=512)

    def forward(self, x):
        x = self.network(x)
        x = self.linear(x.reshape(x.shape[0], -1))
        x = F.sigmoid(x)
        return x


class Audio(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(kernel_size=3, padding=1, in_channels=1, out_channels=4),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=4),
                                     nn.MaxPool2d(kernel_size=2),
                                     nn.Conv2d(kernel_size=3, padding=1, in_channels=4, out_channels=8),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=8),
                                     nn.MaxPool2d(kernel_size=2),
                                     nn.Conv2d(kernel_size=3, padding=1, in_channels=8, out_channels=12),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=12),
                                     nn.MaxPool2d(kernel_size=2))
        self.linear = nn.Linear(in_features=4320, out_features=128)

    def forward(self, x):
        x = self.network(x)
        x = self.linear(x.reshape(x.shape[0], -1))
        x = F.sigmoid(x)
        return x


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.video = Video()
        self.audio = Audio()
        self.linear = nn.Linear(in_features=640, out_features=7)

    def forward(self, audio, video):
        video = self.video(video)
        audio = self.audio(audio)
        x = torch.cat([audio, video], dim=1)
        x = self.linear(x)
        return x
