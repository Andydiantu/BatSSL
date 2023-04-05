import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from matplotlib import patches
from matplotlib.collections import PatchCollection
import pandas as pd

from tqdm import tqdm # TO SHOW THE PROGRESS BAR
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim 


from dfLoader import getDataLoaderSplit, classes, genDataLoaderFromDataset, genTrainTestLoader
from resNet18LinEval import ResNet_18, ResNet_18_linEval
from matplotlib.pyplot import figure
from evalModel import evalModelNew, save_plots, evalModelPerEpoch

import os

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
# import debugpy
import argparse

# debugpy.listen(5678)
# print("waiting for debugger!")
# debugpy.wait_for_client()
# print('Attached')

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
# parser.add_argument('--PercentOfLabelData', type=float, default=1.0, metavar='N',
#                     help='protion of the labbeld data used to fine-tune (default 1.0)')
parser.add_argument('--datasetPath', type=str, default='', metavar='N',
                    help='path of the dataset used for traning')
parser.add_argument('--numData', type=int, default=302, metavar='N',
                    help='number of data for each class to train (default: 302)')

args = parser.parse_args()

print(args.epochs)
# print(args.PercentOfLabelData)
print(args.numData)
print(args.datasetPath)





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

def evalModel(model, testLoader):

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images.view(-1,1, 128, 32))
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

####################### Training CODE #################################

EPOCHS = args.epochs
learning_rate = 0.001
scheduler_patience = 5
batch_size = 32

trainLoader = genDataLoaderFromDataset(datasetPath = args.datasetPath, dataAug = True, augAmount = 4, batchSize = 64)
validLoader = genDataLoaderFromDataset(datasetPath = 'src/datasets/Valid_dataSet.pt', dataAug = False, augAmount = 0, batchSize = 32)
testLoader = genDataLoaderFromDataset(datasetPath = 'src/datasets/Test_dataSet.pt', dataAug = False, augAmount = 0, batchSize = 32)

# # trainLoader = getDataLoaderSplit(batchSize=batch_size,ifAug=True, ifTrain=True,splitPercent = args.PercentOfLabelData)
# testLoader = getDataLoaderSplit(batchSize=32, ifAug=False, ifTrain=False, splitPercent =1.0)

# trainLoader, testLoader = genTrainTestLoader(batchSize = 32, dataAug = True, dataCount = args.numData, testAmount = 0.25)

net = ResNet_18(1,18).to(device)
criterion = nn.CrossEntropyLoss().to(device)
# optimizer = optim.Adam(net.parameters(), lr=0.001)
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay = 1e-3)
# lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience, verbose=True)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * len(trainLoader))

train_acc = []
train_loss = []
valid_acc = []
valid_loss = []
best_acc = 0.0

for epoch in range(EPOCHS):
    losses = []
    train_running_correct = 0
    for data in trainLoader:
        X,y = data
        y = torch.tensor(y, dtype=torch.long).to(device)
        net.zero_grad()
        # print(type(X))
        # print(X.shape)
        output = net(X.view(-1,1, 128, 32).to(device))
        # print(output)
        # print(y)

        _, preds = torch.max(output.data, 1)
        train_running_correct += (preds == y).sum().item()

        loss = criterion(output, y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()


    mean_loss = sum(losses)/len(losses)
    print(f'loss for this epoch {epoch + 1} is {mean_loss}')
    lr_scheduler.step(mean_loss)

    epoch_acc = 100. * (train_running_correct / len(trainLoader.dataset))
    eval_acc, eval_loss = evalModelPerEpoch(net, validLoader, classes, criterion)

    valid_acc.append(eval_acc)
    valid_loss.append(eval_loss)
    train_acc.append(epoch_acc)
    train_loss.append(mean_loss)

    if eval_acc > best_acc:
        best_acc = eval_acc
        torch.save(net.state_dict(), 'src/models/bestSupervised.pth')

net.load_state_dict(torch.load('src/models/bestSupervised.pth'))
trueLabels, predictLabels = evalModelNew(net, testLoader, classes)
save_plots(train_acc, train_loss, valid_acc, valid_loss, f'Baseline_{(args.datasetPath)[13:]}data_{args.epochs}epochs')


