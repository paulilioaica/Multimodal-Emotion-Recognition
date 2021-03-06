import os
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import librosa
from PIL import Image
import random


class FG2020_test(Dataset):
    def __init__(self, root_dir, audio_dir, video_dir, kinect_dir, k_fold_list, transforms=None):
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.kinect_dir = kinect_dir
        self.root_dir = root_dir
        self.k_fold_list = k_fold_list
        self.transforms = transforms
        self.eval_indx = {}
        assert os.path.exists(os.path.join(self.root_dir, self.video_dir)), "Path to videos cannot be found"
        assert os.path.exists(os.path.join(self.root_dir, self.audio_dir)), "Path to audio files cannot be found"
        self.videos = sorted(
            [os.path.join(self.root_dir, self.video_dir, file) for file in self.k_fold_list if file.endswith(".npz")])
        self.audio = sorted(
            [os.path.join(self.root_dir, self.audio_dir, file) for file in self.k_fold_list if file.endswith(".npz")])
        self.kinect = sorted(
            [os.path.join(self.root_dir, self.kinect_dir, file) for file in self.k_fold_list if file.endswith(".npz")])

    def __len__(self):
        return len(self.videos)

    def modify(self, image, seed):
        random.seed(seed)
        rand_n = random.uniform(0, 1)
        image = TF.to_pil_image(image)
        if rand_n > 0.5 and self.transforms:
            image = TF.hflip(image)
            angle = transforms.RandomRotation.get_params((-20, 20))
            image = TF.rotate(image, angle)
        if rand_n > 0.3:
            w, h = image.size
            start, end = transforms.RandomPerspective.get_params(w, h, 0.2)
            image = TF.perspective(image, start, end, interpolation=Image.BICUBIC)

        image = TF.to_tensor(image)
        mean = [image[i, :, :].mean() / 255 for i in range(3)]
        image = TF.normalize(image, mean=mean, std=[1, 1, 1])
        return image

    def transform(self, video, seed):
        video_ret = torch.zeros((video.shape[0], video.shape[3], video.shape[1], video.shape[2]))
        for i, pic in enumerate(video):
            video_ret[i, :, :, :] = self.modify(pic, seed)
        return video_ret

    def __getitem__(self, item):
        seed = np.random.randint(2147483647)
        num_segments = 9
        video = np.load(self.videos[item])['arr_0']
        audio = np.load(self.audio[item])['arr_0'][:, 0]
        kinect = np.load(self.kinect[item])['arr_0']
        video_idx = []
        kinect_idx = []
        audio_slices = []
        for i in np.linspace(0, video.shape[0]-1, num_segments):
            indx = int(i)
            video_idx.append(indx)
        for i in np.linspace(0, kinect.shape[0]-1, num_segments):
            indx = int(i)
            kinect_idx.append(indx)

        spectrogram = librosa.feature.melspectrogram(audio)
        audio = np.array(Image.fromarray(spectrogram).resize((100,150), Image.ANTIALIAS))[np.newaxis, :]
        video = video[video_idx]
        kinect = kinect[kinect_idx]
        label = int(self.videos[item].split('.')[-2])
        video = self.transform(video, seed)
        return video, audio, kinect, label
