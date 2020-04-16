


import numpy as np
import torch.nn.functional as F

import torch
from analytics import Confusion
margin = 2
def loss_contrastive(output1, output2, label):
    label = label.cpu()
    euclidean_distance = F.pairwise_distance(output1, output2)
    euclidean_distance = euclidean_distance.cpu()
    loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                  (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss_contrastive

class Trainer:
    def __init__(self, network, train_dataloader, eval_dataloader, criterion, optimizer,
                 config, device):
        self.device = device
        self.config = config
        self.network = network
        self.confusion = Confusion((7, 7))
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.criterion = criterion
        self.optimizer = optimizer

    def train_epoch(self, epoch, total_epoch):
        running_loss = []
        accuracy = []
        for idx, (input1, input2, label) in enumerate(self.train_dataloader, 0):
            self.optimizer.zero_grad()
            video_1, audio_1, kinect_1 = input1
            video_2, audio_2, kinect_2 = input2

            kinect_1 = kinect_1.permute(0, 1, 4, 2, 3)
            kinect_2 = kinect_2.permute(0, 1, 4, 2, 3)
            out1, out2 = self.network((video_1.to(self.device).float(), audio_1.to(self.device).float(), kinect_1.to(self.device).float()),
                                  (video_2.to(self.device).float(), audio_2.to(self.device).float(), kinect_2.to(self.device).float()))
            loss = loss_contrastive(out1, out2, label)
            loss.backward()
            self.optimizer.step()
            running_loss.append(loss.item())
        print("[TRAIN] Epoch {}/{},  Loss is {}".format(epoch, total_epoch, np.mean(running_loss)))
        return 0, np.mean(running_loss)

    def eval_net(self):
        running_eval_loss = []
        self.network.eval()
        accuracy = []
        support = self.train_dataloader.dataset.get_support()
        batch_predictions = []
        for idx, (input1, label) in enumerate(self.eval_dataloader, 0):
            video_1, audio_1, kinect_1 = input1
            kinect_1 = kinect_1.permute(0, 1, 4, 2, 3)
            min_distance = 99999
            predicted_label = None
            for supp in support:
                video_2, audio_2, kinect_2, local_label = supp
                video_2 = torch.tensor(video_2).permute(0,3,1,2).unsqueeze(0)
                audio_2 = torch.tensor(audio_2).unsqueeze(0)
                kinect_2 = torch.tensor(kinect_2).unsqueeze(0).permute(0, 1, 4, 2, 3)
                output1, output2 = self.network(
                    (video_1.to(self.device).float(), audio_1.to(self.device).float(), kinect_1.to(self.device).float()),
                    (video_2.to(self.device).float(), audio_2.to(self.device).float(), kinect_2.to(self.device).float()))
                euclidean_distance = F.pairwise_distance(output1, output2)
                if euclidean_distance.item() < min_distance:
                    min_distance = euclidean_distance.item()
                    predicted_label = local_label
            batch_predictions.append(predicted_label)
            print(predicted_label)
        accuracy = sum([1 for i in range(input1.shape[0]) if batch_predictions[i] == label[i]]) /(input1.shape[0])



    def train(self):
        training_loss = []
        validation_loss = []

        training_accuracy = []
        validation_accuracy = []
        for i in range(1, self.config['epochs'] + 1):
            self.eval_net()

        return training_accuracy, training_loss, validation_accuracy, validation_loss
