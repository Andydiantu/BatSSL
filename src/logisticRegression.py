import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
import argparse
import torch
from evalModel import printClassAccuracy
from dfLoader import classes
import torch.nn as nn

t2np = lambda t: t.detach().cpu().numpy()

def init_weights(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
    elif isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)


parser = argparse.ArgumentParser()
parser.add_argument('--modelOutputName', type=str, default='Autoencoder', metavar='N',
                    help='Name of model file used for finetunning')
parser.add_argument('--modelType', type=str, default='Autoencoder', metavar='N',
                    help='Autoencoder or SimCLR')
parser.add_argument('--datasetPath', type=str, default='', metavar='N',
                    help='path of the dataset used for traning')

args = parser.parse_args()
print(args.datasetPath)
print(args.modelOutputName)
print(args.modelType)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(device)



dataset = torch.load(args.datasetPath)
X_train = dataset.tensors[0].to(device)
y_train = dataset.tensors[1].to(device)

test_dataset = torch.load('src/datasets/Test_dataSet.pt')
X_test = test_dataset.tensors[0].to(device)
y_test = test_dataset.tensors[1].to(device)



if(args.modelType == 'Autoencoder'):

    from autoEncoder import ConvAutoencoder, DSModel

    
    inputChannel = 1
    latent_size = 512
    linEval = True

    model = ConvAutoencoder(inputChannel, latent_size)
    model.load_state_dict(torch.load(f'src/models/{args.modelOutputName}.pth'))
    model.to(device)

    DSmodel = DSModel(model,18, latent_size, linEval).to(device)

elif(args.modelType == 'SimCLR'):

    from simCLR import SimCLR, DSModel

    inputChannel = 1
    latent_size = 512
    linEval = True

    model = SimCLR(inputChannel, latent_size)
    model.load_state_dict(torch.load(f'src/models/{args.modelOutputName}.pth'))
    model.to(device)

    DSmodel = DSModel(model,18, linEval, latent_size).to(device)

else:

    from resNet18LinEval import ResNet_18, ResNet_18_linEval

    latent_size = 512
    model = ResNet_18(1,18)
    # model.load_state_dict(torch.load(f'src/models/bestSupervised.pth'))
    model.to(device)

    DSmodel = ResNet_18_linEval(model,latent_size, 18).to(device)

    # DSmodel = ResNet_18(1,512)
    # DSmodel.to(device)

    for param in DSmodel.parameters():
        init_weights(param)



X_train_representation = DSmodel(X_train.view(-1,1,128,32))
X_test_representation = DSmodel(X_test.view(-1,1,128,32))

X_train_representation = t2np(X_train_representation)
X_test_representation  = t2np(X_test_representation)

y_train = t2np(y_train)
y_test = t2np(y_test)

lm = linear_model.LogisticRegression()
lm.fit(X_train_representation, y_train)

predicted = lm.predict(X_test_representation)

print()
print()
print(metrics.classification_report(y_test, predicted))
print()
print()

printClassAccuracy(classes, y_test,predicted)



