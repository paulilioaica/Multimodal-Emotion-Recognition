import json
import os

import numpy as np

from dataloader import CremaD
from main import run
import matplotlib.pyplot as plt

with open('config.json', 'r') as f:
    config = json.load(f)

unique_individuals = list(
    set([file.split(".")[0] for file in os.listdir(os.path.join(config['path'], config['video']))]))
individuals = {individ: [file for file in os.listdir(os.path.join(config['path'], config['video'])) if individ in file.split(".")[0]]
               for individ in unique_individuals}

total_size = len(individuals)
bucket_size = 2
k_buckets = 3

buckets = []

for bucket in range(0, bucket_size * (k_buckets - 1) + 1, bucket_size):
    end = bucket + bucket_size
    if bucket == bucket_size * (k_buckets - 1) and total_size - (bucket + bucket_size) > 0:
        end = total_size
    aux = []
    for i in range(bucket, end):
        aux.append(unique_individuals[i])
    buckets.append(aux)

global_acc = []
global_loss = []
bucket_accuracy = []
for i in range(len(buckets)):
    train = []
    test = []
    for j, bucket in enumerate(buckets):
        if i != j:
            for idx in bucket:
                train.extend(individuals[idx])
        else:
            for idx in bucket:
                test.extend(individuals[idx])

    print(train)
    print(test)
    train_dataset = CremaD(config['path'], config['audio'], config['video'], k_fold_list=train)
    val_dataset = CremaD(config['path'], config['audio'], config['video'], k_fold_list=test)
    train_acc, train_loss, val_acc, val_loss = run(config, train_dataset, val_dataset)
    bucket_accuracy.append([train_acc, train_loss, val_acc, val_loss])
    print("Tested on bucket number {}\n loss:train {}, test{} \n acc: train {}, test{}".format(i, train_loss, val_loss,
                                                                                               train_acc, val_acc))

#
#
# plt.title("Loss")
# plt.plot(x, train_loss)
# plt.plot(x, val_loss)
# plt.show()
#
# plt.title("Acuratetea")
# plt.plot(x, train_acc)
# plt.plot(x, val_acc)
# plt.show()
