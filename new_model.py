import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 128
        self.gru = nn.GRUCell(input_size=512, hidden_size=self.hidden_size)

    def forward(self, x, h):
        x = self.gru(x, h)
        return x


class Video(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(*list(resnet18(pretrained=True).children())[:-2])
        for params in self.network.parameters():
            params.requires_grad = False
        self.linear = nn.Linear(in_features=25088, out_features=512)

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
        self.gru = GRU()
        self.linear = nn.Linear(in_features=256, out_features=7)

    def forward(self, audio, video_arr):
        if torch.cuda.is_available():
            h = torch.zeros((video_arr.shape[0], 128)).cuda()
        else:
            h = torch.zeros((video_arr.shape[0], 128))
        for i in range(video_arr.shape[1]):
            video_seq = self.video(video_arr[:, i, :, :])
            h = self.gru(video_seq, h)
        audio = self.audio(audio)
        x = torch.cat([audio, h], dim=1)
        x = self.linear(x)
        return x
