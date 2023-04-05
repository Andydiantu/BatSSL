import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Function
from torch.autograd import Variable
import math
from resNet18 import ResNet_18
from dataAug import augDataSingle, genAugData
from pytorch_metric_learning.losses import NTXentLoss
from dfLoader import getDataLoader, classes,getDataLoaderSplit, genDataLoaderFromDataset
from evalModel import evalModelNew, contrastAccEval, save_plots
from simCLR import SimCLR, DSModel

import torch.optim as optim 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs_train', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--modelOutputName', type=str, default='SimCLR', metavar='N',
                    help='Name of output model file')

args = parser.parse_args()

print(args.epochs_train)
print(args.modelOutputName)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)





####################### Training CODE #################################


loss_termperature = 0.5
learning_rate = 0.001
EPOCHS = args.epochs_train
latent_size = 512
scheduler_patience = 5
batch_size = 128

train_loader = genDataLoaderFromDataset(datasetPath = 'src/datasets/Unlabbeled_train_dataSet.pt', dataAug = False, augAmount = 0, batchSize = batch_size)


loss_function = NTXentLoss(temperature=loss_termperature)
model = SimCLR(1,latent_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-3)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience, verbose=True)
# lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * len(train_loader))


train_acc = []
train_loss = []
for epoch in range(EPOCHS):
    losses = []
    train_running_correct = 0
    for data in train_loader:
        x,_ = data
        
        # x = x.view(-1,1,128,32)

        optimizer.zero_grad()
        # TODO need to be changed
        # print(f'shape for the x is {x.shape}')
        x1 = genAugData(torch.clone(x),1).view(-1,1,128,32)
        x2 = genAugData(torch.clone(x),1).view(-1,1,128,32)
        
        
        x1 = x1.to(device)
        x2 = x2.to(device)
        
        # print(f'shape of data is {x1.size()} and {x2.shape}')

        x1 = model(x1)
        x2 = model(x2)

        # print(f'the data looks like this {x1}')
        

        # print(f'shape of data is {x1.size()} and {x2.size()}')

        embeddings = torch.cat((x1,x2))
        indices = torch.arange(0, x1.size(0), device=x1.device)
        labels = torch.cat((indices,indices))

        # print(f'shape of labels is {labels.size()}')
        # print(f'shape of embeddings is {embeddings.size()}')

        loss = loss_function(embeddings, labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        
        train_running_correct += contrastAccEval(embeddings, labels)
        

    mean_loss = sum(losses)/len(losses)
    print(f'loss for this epoch {epoch + 1} is {mean_loss}')
    lr_scheduler.step(mean_loss)
    epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))
    train_acc.append(epoch_acc)
    train_loss.append(mean_loss)

save_plots(train_acc, train_loss, train_acc, train_loss, f'Contrast_{args.epochs_train}epochs')
torch.save(model.state_dict(), f'src/models/{args.modelOutputName}.pth')


        
        
