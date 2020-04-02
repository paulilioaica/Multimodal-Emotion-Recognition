import torch.nn as nn
import torch.nn.functional as F
import torch


class Video(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(kernel_size=5, padding=0, in_channels=1, out_channels=16, stride=2),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=16),
                                     nn.MaxPool2d(kernel_size=2),
                                     nn.Conv2d(kernel_size=5, padding=0, in_channels=16, out_channels=32),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=32),
                                     nn.MaxPool2d(kernel_size=2),
                                    nn.Conv2d(kernel_size=5, padding=0, in_channels=32, out_channels=32),
                                    nn.ReLU(),
                                     nn.BatchNorm2d(num_features=32),
                                     nn.MaxPool2d(kernel_size=2))
        self.linear = nn.Linear(in_features=2880, out_features=512)

    def forward(self, x):
        x = self.network(x)
        x = self.linear(x.reshape(x.shape[0], -1))
        x = F.sigmoid(x)
        return x


class Audio(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(kernel_size=4, padding=0, in_channels=1, out_channels=8, stride=2),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=8),
                                     nn.MaxPool2d(kernel_size=2),
                                     nn.Conv2d(kernel_size=4, padding=0, in_channels=8, out_channels=16),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=16),
                                     nn.MaxPool2d(kernel_size=2))
        self.linear = nn.Linear(in_features=4576, out_features=128)

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
