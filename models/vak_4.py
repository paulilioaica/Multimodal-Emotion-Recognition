import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision.models import resnet18





class Resnet():
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(*list(resnet18(pretrained=True).children())[:-1])
        for params in self.network.parameters():
            params.requires_grad = False


class Kinect(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = Resnet().network

    def forward(self, x):
        x = self.network(x)
        return x.reshape(x.shape[0], -1)


class Video(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = Resnet().network

    def forward(self, x):
        x = self.network(x)
        return x.reshape(x.shape[0], -1)


class Audio(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(kernel_size=4, padding=0, in_channels=1, out_channels=16, stride=2, bias=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(kernel_size=4, padding=0, in_channels=16, out_channels=8, bias=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=8),
            nn.MaxPool2d(kernel_size=2))
        self.linear = nn.Linear(in_features=1360, out_features=128)

    def forward(self, x):
        x = self.network(x)
        x = self.linear(x.reshape(x.shape[0], -1))
        x = F.relu(x)
        return x


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.video = Video()
        self.audio = Audio()
        self.kinect = Kinect()
        self.linear = nn.Linear(in_features=1152, out_features=7)

    def forward(self, audio_arr, video_arr, motion_arr):
        x = []
        audio = self.audio(audio_arr)
        for i in range(motion_arr.shape[1]):
            video_seq = self.video(video_arr[:, i, :, :, :])
            kinect_seq = self.kinect(motion_arr[:, i, :, :, :].transpose(1, 3))
            x.append(torch.cat([video_seq, kinect_seq], dim=1))
            mean = torch.mean(torch.stack(x, dim=1), dim=1)
            result = torch.cat([audio,mean],dim=1)
        x = self.linear(result)
        return x
