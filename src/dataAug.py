import numpy as np
import torch
from skimage.transform import rotate, AffineTransform, warp
import torchaudio.transforms as T
import torchvision.transforms as VT
import cv2
import random

def freqMasking(spec, freq_mask_param):

    masking = T.FrequencyMasking(freq_mask_param)
    spec = masking(spec)    
    return spec

def timeMasking(spec, time_mask_param):
    spec = spec[np.newaxis, :]
    
    masking = T.TimeMasking(time_mask_param=time_mask_param)
    spec = masking(spec)
    return spec[0]

def timeWarping(spec, wrap_param):
    transform = AffineTransform(translation=(wrap_param,0))
    wrapShift = warp(spec,transform,mode='wrap')
    
    return  wrapShift


def randomStretch(spec, strechRange):
    finalWidth = int(spec.shape[1] * (1+strechRange))
    orgShape = spec.shape
    spec = cv2.resize(spec.numpy(),(finalWidth,128))
    startPoint = random.randint(0, int(orgShape[1]*strechRange))
    spec = spec[0:orgShape[0],startPoint:startPoint+32]
    spec = torch.from_numpy(spec)
    return spec

def batchRandomStretch(specList,strechRange ):
    results = []
    for i, x in enumerate(specList):
        stretchedSpec = randomStretch(x.view(128,32),strechRange)
        results.append(stretchedSpec.view(1,128,32))

    return torch.stack(results)



def augDataSingle(data):
    # print(data.shape)
    aug_spec = timeMasking(data, 10)
    # print(aug_spec.shape)
    aug_spec = freqMasking(aug_spec, 40)
    # print(aug_spec.shape)
    aug_spec = batchRandomStretch(aug_spec,0.2)
    # print(aug_spec.shape)
    blurrer = VT.GaussianBlur(kernel_size=(5, 5), sigma=(0.1,1.0))
    # print(aug_spec.shape)
    aug_spec = blurrer(aug_spec)
    return aug_spec

def genAugData(data, amount):

    augData = []
    
    # print(f' the length of input data is {len(data)}')

    blurrer = VT.GaussianBlur(kernel_size=(5, 5), sigma=(0.1,1.0))

    for j in range(amount):

        for i in range(len(data)):
            aug_spec = data[i]

            if(bool(random.getrandbits(1))):
                aug_spec = timeMasking(aug_spec, 10)
            if(bool(random.getrandbits(1))):
                aug_spec = freqMasking(aug_spec, 30)
            if(bool(random.getrandbits(1))):
                aug_spec = VT.RandomErasing(p=1,scale=(0.10, 0.40),ratio=(0.3, 0.9))(aug_spec.view(-1,128,32)).view(128,32)

            if(bool(random.getrandbits(1))):
                aug_spec = torch.roll(aug_spec,random.randint(0,15), dims = 1)
            if(bool(random.getrandbits(1))):
                if(bool(random.getrandbits(1))):
                    aug_spec = torch.roll(aug_spec,random.randint(0,7), dims = 0)
                else:
                    aug_spec = torch.roll(aug_spec,-random.randint(0,7), dims = 0)

            if(bool(random.getrandbits(1))):
                aug_spec = randomStretch(aug_spec,0.2)

            if(bool(random.getrandbits(1))):
                aug_spec = blurrer(aug_spec.view(1,128,32)).view(128,32)

            
            augData.append(aug_spec)

    augData = torch.stack(augData)


    return augData

