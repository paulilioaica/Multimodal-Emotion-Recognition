import numpy as np


class Confusion:
    def __init__(self, shape):
        self.shape = shape
        self.matrix = np.zeros(shape)

    def update(self, guess, target):
        target = target.detach().cpu()
        guess = guess.cpu()
        for i in range(guess.shape[0]):
            self.matrix[np.argmax(guess[i])][target[i]] += 1

    def print(self):
        print(self.matrix)
        self.matrix = np.zeros(self.shape)
