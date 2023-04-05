from simCLR import SimCLR, DSModel
from dfLoader import getDataLoader, classes,getDataLoaderSplit, genDataLoaderFromDataset
from evalModel import evalModelNew, contrastAccEval, save_plots, evalModelPerEpoch
import torch.optim as optim 
import argparse
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--epochs_Finetune', type=int, default=10, metavar='N',
                    help='number of epochs to finetune (default: 10)')
parser.add_argument('--linEval', type=str, default='False', metavar='N',
                    help='True to use linear Evalution, fine tunning otherwise')
parser.add_argument('--datasetPath', type=str, default='', metavar='N',
                    help='path of the dataset used for traning')
parser.add_argument('--modelOutputName', type=str, default='SimCLR', metavar='N',
                    help='Name of model file used for finetunning')


args = parser.parse_args()

print(args.epochs_Finetune)
print(args.datasetPath)
print(args.modelOutputName)
if(args.linEval == 'False'):
    linEval = False
else:
    linEval = True
print(linEval)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



############################### Fine Tunning ################################

learning_rate = 0.001
EPOCHS = args.epochs_Finetune
latent_size = 512
scheduler_patience = 5
# train_loader = getDataLoaderSplit(128,False,True,args.PercentOfLabelData)

train_loader = genDataLoaderFromDataset(datasetPath = args.datasetPath, dataAug = True, augAmount = 4, batchSize = 64)
valid_loader = genDataLoaderFromDataset(datasetPath = 'src/datasets/Valid_dataSet.pt', dataAug = False, augAmount = 0, batchSize = 32)
test_loader = genDataLoaderFromDataset(datasetPath = 'src/datasets/Test_dataSet.pt', dataAug = False, augAmount = 0, batchSize = 32)

model = SimCLR(1,latent_size)
model.load_state_dict(torch.load(f'src/models/{args.modelOutputName}.pth'))
model.to(device)

DSmodel = DSModel(model,18,linEval, latent_size).to(device)
optimizer = torch.optim.Adam(DSmodel.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss().to(device)
# lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience, verbose=True)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS/2, eta_min = 1e-08)

train_acc = []
train_loss = []
valid_acc = []
valid_loss = []
best_acc = 0.0


for epoch in range(EPOCHS):
    losses = []
    train_running_correct = 0
    for data in train_loader:
        
        x,y = data

        x = x.view(-1,1,128,32)

        x = x.to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)
        
        outputs = DSmodel(x)

        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == y).sum().item()

        loss = criterion(outputs,y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))
    mean_loss = sum(losses)/len(losses)
    eval_acc, eval_loss = evalModelPerEpoch(DSmodel, valid_loader, classes, criterion)
    print(f'loss for this epoch {epoch + 1} is {mean_loss}')
    lr_scheduler.step(mean_loss)

    valid_acc.append(eval_acc)
    valid_loss.append(eval_loss)
    train_acc.append(epoch_acc)
    train_loss.append(mean_loss)

    if eval_acc > best_acc:
        best_acc = eval_acc
        torch.save(DSmodel.state_dict(), 'src/models/bestSimCLRFinetune.pth')
    
DSmodel.load_state_dict(torch.load('src/models/bestSimCLRFinetune.pth')) 
trueLabels, predictLabels = evalModelNew(DSmodel, test_loader,classes)
save_plots(train_acc, train_loss, valid_acc, valid_loss, f'SimCLR_{(args.datasetPath)[13:]}_data_{args.epochs_Finetune}epochs')

        
        
