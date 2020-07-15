import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18


class Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(*list(r3d_18(pretrained=True).children())[:-1])
        for params in self.network.parameters():
            params.requires_grad = False
        print(self.network)

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
        self.resnet = Resnet()
        self.audio = Audio()
        self.linear = nn.Linear(in_features=1024+128, out_features=7)

    def forward(self, audio_arr, video_arr, motion_arr):
        motion_arr = motion_arr.transpose(2, 4)
        video_arr = self.resnet(video_arr.transpose(1, 2))
        kinect_arr = self.resnet(motion_arr.transpose(1,2))
        audio = self.audio(audio_arr)
        x = torch.cat([audio, video_arr, kinect_arr], dim=1)
        x = self.linear(x)
        return x