import numpy as np


class Confusion:
    def __init__(self, shape):
        self.matrix = np.zeros(shape)

    def update(self, guess, target):
        for i in range(guess.shape[0]):
            target = target[i]
            guess = np.argmax(guess[i])
            self.matrix[guess][target] += 1

    def print(self):
        print(self.matrix)
