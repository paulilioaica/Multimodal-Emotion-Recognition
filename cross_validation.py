import json
import os
from dataloader import CremaD
from main import run

with open('config.json', 'r') as f:
    config = json.load(f)

unique_individuals = list(
    set([file.split(".")[0] for file in os.listdir(os.path.join(config['path'], config['video']))]))
individuals = {individ: [file for file in os.listdir(os.path.join(config['path'], config['video'])) if individ in file.split(".")[0]]
               for individ in unique_individuals}

total_size = len(individuals)

global_acc = []
global_loss = []
bucket_accuracy = []
for i in range(len(unique_individuals) - 1):
    test = sum([individuals[x] for x in [unique_individuals[i], unique_individuals[i+1]]], [])
    train = sum([individuals[x] for x in unique_individuals[:i] + unique_individuals[i+2:]], [])


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
