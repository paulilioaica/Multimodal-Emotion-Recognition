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



class Kinect(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(*list(resnet18(pretrained=True).children())[:-1])
        for params in self.network.parameters():
            params.requires_grad = False


    def forward(self, x):
        x = self.network(x)
        return x.reshape(x.shape[0], -1)


class Video(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(*list(resnet18(pretrained=True).children())[:-1])
        for params in self.network.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.network(x)
        return x.reshape(x.shape[0], -1)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=384, out_features=384)
    def forward(self, x):
        x = self.linear(x)
        return F.softmax(x)


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
        self.linear = nn.Linear(in_features=2288, out_features=128)

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
        self.gru_video = GRU()
        self.gru_kinect = GRU()
        self.attn = Attention()
        self.kinect = Kinect()
        self.linear = nn.Linear(in_features=384, out_features=7)

    def forward(self, audio, video_arr, motion_arr):
        if torch.cuda.is_available():
            h_video = torch.zeros((video_arr.shape[0], self.gru_video.hidden_size)).cuda()
            h_kinect = torch.zeros((video_arr.shape[0], self.gru_video.hidden_size)).cuda()
        else:
            h_video = torch.zeros((video_arr.shape[0], self.gru_video.hidden_size))
            h_kinect = torch.zeros((video_arr.shape[0], self.gru_kinect.hidden_size))
        for i in range(video_arr.shape[1]):
            video_seq = self.video(video_arr[:, i, :, :])
            h_video = self.gru_video(video_seq, h_video)
        for i in range(motion_arr.shape[1]):
            kinect_seq = self.kinect(motion_arr[:, i, :, :].transpose(1,3))
            h_kinect = self.gru_kinect(kinect_seq, h_kinect)
        audio = self.audio(audio)
        x = torch.cat([audio, h_video, h_kinect], dim=1)
        weights = self.attn(x)
        x = x * weights
        x = self.linear(x)
        return x
