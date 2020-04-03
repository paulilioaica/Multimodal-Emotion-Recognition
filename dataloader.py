import os
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import librosa
from PIL import Image
import random


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
            [os.path.join(self.root_dir, self.audio_dir, file.split("_")[0]+"npz") for file in self.k_fold_list if file.endswith(".npz")])

    def __len__(self):
        return len(self.videos)

    def modify(self, image, seed):
        random.seed(seed)
        rand_n = random.uniform(0, 1)
        image = TF.to_pil_image(image)
        image = TF.to_grayscale(image, num_output_channels=3)
        if rand_n > 0.5 and self.transforms:
            image = TF.hflip(image)
            w, h = image.size
            start, end = transforms.RandomPerspective.get_params(w, h, 0.5)
            image = TF.perspective(image, start, end, interpolation=Image.BICUBIC)
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.35], std=[0.35])
        return image

    def transform(self, video, seed):
        video_ret = torch.zeros((video.shape[0], video.shape[3], video.shape[1], video.shape[2]))
        for i, pic in enumerate(video):
            video_ret[i, :, :, :] = self.modify(pic, seed)
        return video_ret

    def __getitem__(self, item):
        seed = np.random.randint(2147483647)
        video = np.load(self.videos[item])['arr_0']
        audio = np.load(self.audio[item])['arr_0'][:, 0]
        spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(audio), ref=np.max)
        audio = np.array(Image.fromarray(spectrogram).resize((192, 120), Image.ANTIALIAS))[np.newaxis, :, :]
        label = int(self.videos[item].split('_')[0][-2])
        video = self.transform(video, seed)
        return video, audio, label
