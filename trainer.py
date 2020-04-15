import numpy as np
import torch
from analytics import Confusion


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

            output = self.network((video_1.to(self.device).float(), audio_1.to(self.device).float(), kinect_1.to(self.device).float()),
                                  (video_2.to(self.device).float(), audio_2.to(self.device).float(), kinect_2.to(self.device).float()))
            loss = self.criterion(output, label.to(self.device).float())
            accuracy.append(
                sum([1 for i in range(output.shape[0]) if output[i] == label[i]]) / (output.shape[0]))
            loss.backward()
            self.optimizer.step()
            running_loss.append(loss.item())
        print("[TRAIN] Epoch {}/{}, Accuracy is {}, Loss is {}".format(epoch, total_epoch, np.mean(accuracy),
                                                                       np.mean(running_loss)))
        return np.mean(accuracy), np.mean(running_loss)

    def eval_net(self):
        running_eval_loss = []
        self.network.eval()
        accuracy = []
        for idx, (input1, input2, label) in enumerate(self.train_dataloader, 0):
            video_1, audio_1, kinect_1 = input1
            video_2, audio_2, kinect_2 = input2
            output = self.network(
                (video_1.to(self.device).float(), audio_1.to(self.device).float(), kinect_1.to(self.device).float()),
                (video_2.to(self.device).float(), audio_2.to(self.device).float(), kinect_2.to(self.device).float()))
            loss = self.criterion(output, label.to(self.device).float())
            self.confusion.update(output, label)
            predictions = torch.argmax(output, dim=1)
            accuracy.append(sum([1 for i in range(predictions.shape[0]) if predictions[i] == label[i]]) / \
                            (predictions.shape[0]))
            running_eval_loss.append(loss.item())
        print("[EVAL] Accuracy is {}, Loss is {}".format(np.mean(accuracy), np.mean(running_eval_loss)))
        return np.mean(accuracy), np.mean(running_eval_loss)

    def train(self):
        training_loss = []
        validation_loss = []

        training_accuracy = []
        validation_accuracy = []
        for i in range(1, self.config['epochs'] + 1):
            acc, loss = self.train_epoch(i, self.config['epochs'])
            training_loss.append(loss)
            training_accuracy.append(acc)

            acc, loss = self.eval_net()
            validation_loss.append(loss)
            validation_accuracy.append(acc)
        return training_accuracy, training_loss, validation_accuracy, validation_loss
