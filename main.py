from vak_3 import Classifier
import torch
from trainer import Trainer
from torch.utils.data import DataLoader
import torch.optim as optim


def run(config, train_dataset, val_dataset):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Classifier().to(device)
    print("Training on {}, batch_size is {}, lr is {}".format(device, config['batch_size'], config['lr']))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=0.5)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, config, device)
    train_acc, train_loss, val_acc, val_loss = trainer.train()
    return train_acc, train_loss, val_acc, val_loss

