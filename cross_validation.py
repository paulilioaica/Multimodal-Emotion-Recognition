import json
import os
from random import shuffle

from dataloader import FG2020
from dataloader_test import FG2020_Test

from main import run

with open('config.json', 'r') as f:
    config = json.load(f)
forbidden = {}

unique_individuals = list(
    set([file.split(".")[0] for file in os.listdir(os.path.join(config['path'], config['audio']))]))
individuals = {individ: [file for file in os.listdir(os.path.join(config['path'], config['audio'])) if
                         individ in file.split(".")[0] not in forbidden and file.split(".")[2][0] not in forbidden]
               for individ in unique_individuals}

males = [x for x in individuals if "M" in x and x not in forbidden]
females = [x for x in individuals if "F" in x and x not in forbidden]
shuffle(males)
shuffle(females)
test = [("M6", "F6")]
pool = males + females
total_size = len(individuals)
global_acc = []
global_loss = []
bucket_accuracy = []

for i, duo in enumerate(test):
    print(duo)
    test = sum([individuals[x] for x in duo], [])
    aux = [x for x in pool if x not in duo]
    train = sum([individuals[x] for x in aux], [])
    train_dataset = FG2020(config['path'], config['audio'], config['video'], config['kinect'], k_fold_list=train,
                           transforms=True)
    val_dataset = FG2020_Test(config['path'], config['audio'], config['video'], config['kinect'], k_fold_list=test)
    train_acc, train_loss, val_acc, val_loss = run(config, train_dataset, val_dataset)
    bucket_accuracy.append([train_acc, train_loss, val_acc, val_loss])
    print(
        "Tested on bucket number {}\n loss:\ntrain {}\ntest{} \n acc:\ntrain {}\ntest{}".format(i, train_loss, val_loss,
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
