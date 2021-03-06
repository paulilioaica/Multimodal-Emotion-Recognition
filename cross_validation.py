import json
import os
from random import shuffle
from torchvision import transforms
from dataloader import FG2020
from dataloader_test import FG2020_test
from main import run

with open('config.json', 'r') as f:
    config = json.load(f)

unique_individuals = list(
    set([file.split(".")[0] for file in os.listdir(os.path.join(config['path'], config['video']))]))
individuals = {individ: [file for file in os.listdir(os.path.join(config['path'], config['video'])) if
                         individ in file.split(".")[0]]
               for individ in unique_individuals}
forb = {"M5","F5", "F3"}


males = [x for x in individuals if "M" in x and x not in forb]
females = [x for x in individuals if "F" in x and x not in forb]
shuffle(males)
shuffle(females)


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
    val_dataset = FG2020_test(config['path'], config['audio'], config['video'], config['kinect'], k_fold_list=test)
    train_acc, train_loss, val_acc, val_loss = run(config, train_dataset, val_dataset)
    bucket_accuracy.append([train_acc, train_loss, val_acc, val_loss])
    print(
        "Tested on bucket number {}\n loss:\ntrain {}\ntest{} \n acc:\ntrain {}\ntest{}".format(i, train_loss, val_loss,
                                                                                                train_acc, val_acc))

