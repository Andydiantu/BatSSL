import torch
import torch.nn as nn
import torch.nn.functional as F
from resNet18 import ResNet_18
from dfLoader import getDataLoader,classes,getDataLoaderSplit, genTrainTestLoader, genDataLoaderFromDataset
from evalModel import evalModelNew, save_plots, plot_encoder_result
from autoEncoder import ConvAutoencoder, DSModel
import torch.optim as optim 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs_train', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--modelOutputName', type=str, default='Autoencoder', metavar='N',
                    help='Name of output model file')

args = parser.parse_args()

print(args.epochs_train)
print(args.modelOutputName)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


####################### Training CODE #################################

learning_rate = 0.001
inputChannel = 1
latent_size = 512
EPOCHS = args.epochs_train
scheduler_patience = 5
batch_size = 128

# train_loader = getDataLoader(batch_size,ifAug = True,ifTrain = True)
train_loader = genDataLoaderFromDataset(datasetPath = 'src/datasets/Unlabbeled_train_dataSet.pt', dataAug = False, augAmount = 0, batchSize = batch_size)

print(device)

model = ConvAutoencoder(inputChannel, latent_size).to(device)
# criterion = nn.BCELoss()
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-3)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience, verbose=True)
# lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * len(train_loader))


train_losses = []

for epoch in range(1, EPOCHS+1):
    train_loss = 0.0
    for data in train_loader:
        images, _ = data
        images = images.view(-1,1,128,32).to(device)
        outputs = model(images)
#         print(outputs.shape)
#         print(images.shape)
        loss = criterion(outputs, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()*images.size(0)

    

    if(epoch % 10 == 0):
        test_data = images[0]
        output_test = model(test_data.view(-1,1,128,32))
        test_data = torch.clone(test_data).cpu().detach()
        output_test = torch.clone(output_test).cpu().detach()
        plot_encoder_result(test_data.view(128,32), output_test.view(128,32), f'{epoch} epoch result')
    

    train_loss = train_loss/len(train_loader)
    lr_scheduler.step(train_loss)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))
    train_losses.append(train_loss)

save_plots(train_losses, train_losses,train_losses,train_losses, f'autoEncoder_{args.epochs_train}epochs')
torch.save(model.state_dict(), f'src/models/{args.modelOutputName}.pth')

