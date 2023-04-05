import torch
import torch.nn as nn
import torch.nn.functional as F
from resNet18 import ResNet_18


class SimCLR (nn.Module):
    def __init__(self, image_channels, encoder_output_num):
        super().__init__()

        self.encoder = ResNet_18(image_channels, encoder_output_num)
        self.fstProjHead = nn.Linear(encoder_output_num, encoder_output_num)
        self.projection = nn.Sequential(
            
            nn.ReLU(),
            nn.Linear(encoder_output_num, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.fstProjHead(x)
        x = self.projection(x) 

        
        return x



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class DSModel(nn.Module):
    def __init__(self,simCLR,num_classes, linEval, latent_size):
        super().__init__()
        simCLR.projection = Identity()
        self.simCLREncoder = simCLR
        
 
        if(linEval):
            simCLR.fstProjHead = Identity()

            for p in self.simCLREncoder.parameters():
                # print()
                # print('Doing linear evaluation')
                # print()
                p.requires_grad = False
            
        self.lastlayer = nn.Linear(latent_size,num_classes)
        
    def forward(self,x):
        x = self.simCLREncoder(x)
        x = self.lastlayer(x)
        
        return x