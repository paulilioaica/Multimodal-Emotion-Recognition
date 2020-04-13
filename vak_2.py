import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision.models import resnet18


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        self.gru = nn.LSTMCell(input_size=1152, hidden_size=self.hidden_size)

    def forward(self, x, h):
        x = self.gru(x, h)
        return x

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
        self.lstm = LSTM()
        self.audio = Audio()
        self.kinect = Kinect()
        self.linear = nn.Linear(in_features=768, out_features=7)

    def forward(self, audio, video_arr, motion_arr):
        if torch.cuda.is_available():
            h = torch.rand((video_arr.shape[0], self.lstm.hidden_size)).cuda()
            c = torch.rand((video_arr.shape[0], self.lstm.hidden_size)).cuda()
        else:
            h = torch.rand((video_arr.shape[0], self.lstm.hidden_size))
            c = torch.rand((video_arr.shape[0], self.lstm.hidden_size))
        audio = self.audio(audio)
        for i in range(motion_arr.shape[1]):
            video_seq = self.video(video_arr[:, int(i/2), :, :])
            kinect_seq = self.kinect(motion_arr[:,i, :, :].transpose(1, 3))
            x = torch.cat([audio, video_seq, kinect_seq], dim=1)
            h, c = self.lstm(x, (h, c))
        x = self.linear(h)
        return x
