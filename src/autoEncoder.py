import torch
import torch.nn as nn
import torch.nn.functional as F
from resNet18 import ResNet_18


class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x
class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=1):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)
        self.cov2d  = nn.ConvTranspose2d(512, 512, (4,1), stride=2)
        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=1)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = self.cov2d(x)
        x = F.interpolate(x, scale_factor=4)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        # x = torch.sigmoid(self.conv1(x))
        x = self.conv1(x)
        # print(x.shape)
        x = x.view(x.size(0), 1, 128, 32)
        return x

class Decoder(nn.Module):
    def __init__(self, outputChannel,inputDim):
        super(Decoder, self).__init__()
        self.outputChannel = outputChannel
        self.linear = nn.Linear(inputDim,512)
        self.t_conv1 = nn.ConvTranspose2d(512, 256, (8,2), stride=2)
        self.t_conv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(64, 16, 2, stride=2)
        self.t_conv5 = nn.ConvTranspose2d(16, 1, 2, stride=2)
    def forward(self, z):
        x = self.linear(z)
#         print(x.shape)
        x = x.view(z.size(0),512,1,1)
#         print(x.shape)
#         x = F.interpolate(x, scale_factor=4)
#         print(x.shape)
        x = torch.relu(self.t_conv1(x))
#         print(x.shape)
        x = torch.relu(self.t_conv2(x))
#         print(x.shape)
        x = torch.relu(self.t_conv3(x))
#         print(x.shape)
        x = torch.relu(self.t_conv4(x))
#         print(x.shape)
        x = torch.relu(self.t_conv5(x))
#         print(x.shape)
        x = x.view(x.size(0), self.outputChannel, 128, 32)
        return x


class ConvAutoencoder(nn.Module):
    def __init__(self, inputChannel, lantentSpaceSize):
        super(ConvAutoencoder, self).__init__()
#         ## encoder layers ##
        self.encoder = ResNet_18(inputChannel,lantentSpaceSize)
        self.decoder = ResNet18Dec(z_dim = lantentSpaceSize, nc=inputChannel)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = self.encoder(x)
#         x = x.view(2048,1,1)
#         print(x.shape)
        ## decode ##
        x = self.decoder(x)
        return x

class DSModel(nn.Module):
    def __init__(self,model,num_classes, latent_size, linEval):
        super().__init__()
        
        self.Encoder = model.encoder
        self.num_classes = num_classes
        
        if(linEval):
            for p in self.Encoder.parameters():
                p.requires_grad = False
            
        self.lastlayer = nn.Linear(latent_size,self.num_classes)
        
    def forward(self,x):
        x = self.Encoder(x)
        x = self.lastlayer(x)
        
        return x