import json
import os
from random import shuffle
from torchvision import transforms
from dataloader import CremaD
from main import run

with open('config.json', 'r') as f:
    config = json.load(f)

unique_individuals = list(
    set([file.split(".")[0] for file in os.listdir(os.path.join(config['path'], config['video']))]))
individuals = {individ: [file for file in os.listdir(os.path.join(config['path'], config['video'])) if individ in file.split(".")[0]]
               for individ in unique_individuals}
print(individuals)
total_size = len(individuals)
print(len(unique_individuals))
global_acc = []
global_loss = []
bucket_accuracy = []

trans_train = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.Grayscale(num_output_channels=1), transforms.RandomPerspective(), transforms.ToTensor(), transforms.Normalize(mean=[0.35], std=[0.35])])
trans_eval = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize(mean=[0.4], std=[0.4])])
test = sum([individuals[x] for x in ["F6", "M3"]], [])
individuals.remove("F6")
individuals.remove("M3")
train =sum([individuals[x] for x in unique_individuals], [])

shuffle(test)
shuffle(train)


train_dataset = CremaD(config['path'], config['audio'], config['video'], k_fold_list=train, transforms=True)
val_dataset = CremaD(config['path'], config['audio'], config['video'], k_fold_list=test)
train_acc, train_loss, val_acc, val_loss = run(config, train_dataset, val_dataset)
bucket_accuracy.append([train_acc, train_loss, val_acc, val_loss])
print("Tested on bucket number {}\n loss:\ntrain {}\ntest{} \n acc:\ntrain {}\ntest{}".format(i, train_loss, val_loss,
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
