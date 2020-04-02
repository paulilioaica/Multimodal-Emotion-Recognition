import os
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
from PIL import Image


class CremaD(Dataset):
    def __init__(self, root_dir, audio_dir, video_dir, k_fold_list, transforms=None):
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.root_dir = root_dir
        self.k_fold_list = k_fold_list
        self.transforms = transforms
        assert os.path.exists(os.path.join(self.root_dir, self.video_dir)), "Path to videos cannot be found"
        assert os.path.exists(os.path.join(self.root_dir, self.audio_dir)), "Path to audio files cannot be found"
        self.videos = sorted(
            [os.path.join(self.root_dir, self.video_dir, file) for file in self.k_fold_list if file.endswith(".npz")])
        self.audio = sorted(
            [os.path.join(self.root_dir, self.audio_dir, file) for file in self.k_fold_list if file.endswith(".npz")])

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, item):
        video = np.load(self.videos[item])['arr_0']
        audio = np.load(self.audio[item])['arr_0'][:,0]
        video_frames_num = video.shape[0]
        video_idx = [int(round(i)) for i in np.linspace(0, video_frames_num - 1, num=20)]
        spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(audio), ref=np.max)
        audio = np.array(Image.fromarray(spectrogram).resize((192, 120), Image.ANTIALIAS))[np.newaxis, :, :]
        label = int(self.videos[item].split('.')[-2])
        if self.transforms:
            seq = video[video_idx]
            x = torch.zeros((seq.shape[0], seq.shape[1], seq.shape[2]))
            for i, pic in enumerate(seq):
                x[i,:,:] = self.transforms(pic)
            return x,  audio, label