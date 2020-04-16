import os
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import librosa
from PIL import Image
import random


class FG2020_Test(Dataset):
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
        self.num_classes = 7
        self.classes = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
        for element in k_fold_list:
            label = int(element.split(".")[2])
            self.classes[label].append(element)
        print(self.classes[0])

    def __len__(self):
        return 200

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

        index = random.randint(0, self.num_classes - 1)
        label = index
        first_sample = random.choice(self.classes[index])

        seed = np.random.randint(2147483647)
        video_first = np.load(os.path.join(self.root_dir, self.video_dir, first_sample).split(".npz")[0] + "_0.npz")['arr_0']
        audio_first = np.load(os.path.join(self.root_dir, self.audio_dir, first_sample))['arr_0'][:, 0]
        kinect_first = np.load(os.path.join(self.root_dir,self.kinect_dir, first_sample))['arr_0']
        spectrogram = librosa.feature.melspectrogram(audio_first)
        audio_first = np.array(Image.fromarray(spectrogram).resize((100, 150), Image.ANTIALIAS))[np.newaxis, :]
        video_first = self.transform(video_first, seed)

        return (video_first, audio_first, kinect_first), label