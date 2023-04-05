from doctest import FAIL_FAST
import pandas as pd
import json
import numpy as np
from scipy.io import wavfile
import cv2
import torch
from torch.utils.data import TensorDataset, DataLoader
from dataAug import genAugData
import random
from sklearn.model_selection import train_test_split
import librosa
import torchvision.transforms as transforms

annAudioDic = {
    'BritishBatCalls_MartynCooke_2018_1_sec_train_expert.json':'bat_data_martyn_2018/',
    'BritishBatCalls_MartynCooke_2018_1_sec_test_expert.json':'bat_data_martyn_2018_test/',
    'BritishBatCalls_MartynCooke_2019_1_sec_test_expert.json':'bat_data_martyn_2019_test/',
    'BritishBatCalls_MartynCooke_2019_1_sec_train_expert.json':'bat_data_martyn_2019/',
    'bcireland_expert.json':'bcireland/',
    'sn_scot_nor_0.5_expert.json':'sn_scot_nor/',
    'Echobank_train_expert.json':'echobank/',
    'BCT_1_sec_train_expert.json':'BCT_1_sec/'
}

trainAudioDic = {
    'bcireland_expert.json':'bcireland/',
    'sn_scot_nor_0.5_expert.json':'sn_scot_nor/',
    'Echobank_train_expert.json':'echobank/',
    'BCT_1_sec_train_expert.json':'BCT_1_sec/'
}

testAudioDic = {
    'BritishBatCalls_MartynCooke_2018_1_sec_train_expert.json':'bat_data_martyn_2018/',
    'BritishBatCalls_MartynCooke_2018_1_sec_test_expert.json':'bat_data_martyn_2018_test/',
    'BritishBatCalls_MartynCooke_2019_1_sec_test_expert.json':'bat_data_martyn_2019_test/',
    'BritishBatCalls_MartynCooke_2019_1_sec_train_expert.json':'bat_data_martyn_2019/'
}

classes = ['Myotis daubentonii', 'Plecotus auritus', 'Pipistrellus pipistrellus' , #'Bat',
 'Nyctalus leisleri' ,'Pipistrellus pygmaeus', 'Myotis mystacinus',
 'Myotis nattereri', 'Pipistrellus nathusii', 'Nyctalus noctula',
 'Eptesicus serotinus', 'Barbastellus barbastellus', 'Myotis brandtii',
 'Myotis alcathoe' ,'Myotis bechsteinii', 'Plecotus austriacus',
 'Rhinolophus ferrumequinum', 'Rhinolophus hipposideros','noise']

classesDic = {
    'Myotis daubentonii': 0,
    'Plecotus auritus': 1,
    'Pipistrellus pipistrellus': 2,
    # 'Bat': 3,
    'Nyctalus leisleri': 3,
    'Pipistrellus pygmaeus': 4,
    'Myotis mystacinus': 5,
    'Myotis nattereri': 6,
    'Pipistrellus nathusii': 7,
    'Nyctalus noctula': 8,
    'Eptesicus serotinus': 9,
    'Barbastellus barbastellus': 10,
    'Myotis brandtii': 11,
    'Myotis alcathoe': 12,
    'Myotis bechsteinii': 13,
    'Plecotus austriacus': 14,
    'Rhinolophus ferrumequinum': 15,
    'Rhinolophus hipposideros': 16,
    'noise':17
 }

def genDataLoaderSplit(batchSize, df, dataAug, noiseList, splitPercent):

    X = []
    y = []

    for index, row in df.iterrows():
        reshapeData = row['data']
        X.append(reshapeData)
        y.append(row['label'])

    X.extend(noiseList)
    y.extend([17]*len(noiseList))

    if(splitPercent != 1.0):
        _, X, _, y = train_test_split(X, y, test_size=splitPercent)

    X = torch.Tensor(np.array(X))
    y = torch.Tensor(np.array(y))

    print(f'data size for non bat before aug is {X.size()}')
    print(f'dataAug is {dataAug}')
    if dataAug:
        augData = genAugData(X,3)
        print(f'Data Aug size is {augData.size()}')

        X = torch.concat((X,augData))
        y = torch.concat((y,y,y,y))

    print(f'data size for non bat is {X.size()}')
    print(f'size for label is {y.size()}')
    print()

    datasetTensor = TensorDataset(X,y)

    dataLoader = DataLoader(datasetTensor, batch_size = batchSize, shuffle=True)

    return dataLoader

def genDataLoader(batchSize, df, dataAug, noiseList):

    X = []
    y = []

    for index, row in df.iterrows():
        reshapeData = row['data']
        X.append(reshapeData)
        y.append(0)

    X.extend(noiseList)
    y.extend([0]*len(noiseList))

    X = torch.Tensor(np.array(X))
    y = torch.Tensor(np.array(y))

    print(f'data size for bat before aug is {X.size()}')
    if dataAug:
        augData = genAugData(X,3)
        X = torch.concat((X,augData))
        y = torch.concat((y,y,y,y))
    
    print(f'data size for bat is {X.size()}')
    print(f'size for label is {y.size()}')

    print()


    datasetTensor = TensorDataset(X,y)

    dataLoader = DataLoader(datasetTensor, batch_size = batchSize, shuffle=True)

    return dataLoader

def dfLoader(ann_path, audio_dir, allData):

    # Load annotation and convert to dataframe
    data = json.load(open(ann_path))
    df = pd.json_normalize(data, record_path=['annotation'], meta=['id'])

    if(not allData):
        df = df[df['class'] != 'Bat']

    trimInTime, trimOutTime = calInOutTime(df)
    df['trimInTime'] = trimInTime
    df['trimOutTime'] = trimOutTime

    trimmed_specs, trimmed_specs_PCEN = trimAudio(df, audio_dir)
    df['data'] = trimmed_specs
    df['normData'] = trimmed_specs_PCEN

    specLength = []
    for index, row in df.iterrows():
        specLength.append(row['data'].shape[1])
    df['len'] = specLength

    # Delete data shorter than 20 ms
    df = df[df['len'] ==32]

    if(not allData):
        numLabels = addNumLabels(df)
        df['label'] = numLabels

    resizedSpecs, resizedSpecsPCEN = reSizeSpec(df)
    df['data'] = resizedSpecs
    df['normData'] = resizedSpecsPCEN

    return df
    

def gen_mag_spectrogram(x, nfft, noverlap):
    # Computes magnitude spectrogram by specifying time.

    # window data
    step = nfft - noverlap
    shape = (nfft, (x.shape[-1]-noverlap)//step)
    strides = (x.strides[0], step*x.strides[0])
    x_wins = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # apply window
    x_wins = np.hanning(x_wins.shape[0]).astype(np.float32)[..., np.newaxis] * x_wins

    # do fft
    # note this will be much slower if x_wins.shape[0] is not a power of 2
    complex_spec = np.fft.rfft(x_wins, axis=0)

    # calculate magnitude
    #spec = (np.conjugate(complex_spec) * complex_spec).real
    # same as:
    spec = np.absolute(complex_spec)**2

    # orientate the ocrrect way 
    spec = np.flipud(spec)
  
    # convert to "amplitude"
    spec = np.log(1.0 + spec)

    return spec


def calInOutTime(df):

    # calculate and add timein and timeout
    duration = df['end_time'] - df['start_time']
    df['duration'] = duration

    trimInTime = []
    trimOutTime = []

    for index, row in df.iterrows():
        startTime = row['start_time']
        trimInTime.append(startTime)
        trimOutTime.append(startTime+0.02)
    
    return trimInTime, trimOutTime
            
    



def trimAudio(df, audio_dir):
    trimmed_data = []
    trimmed_data_PCEN = []


    for filename in df['id'].unique():
        fs, x = wavfile.read(audio_dir + filename)
        fileDuration = x.shape[0]/fs
        spec = gen_mag_spectrogram(x, 1024, 840) # TODO do i have to change 840 to 939 ?

        unit = fs/2/spec.shape[0]
        startP = int(10000/unit)
        endP = int(140000/unit) + 1

        spec = spec[startP:endP]

        #TODO do i have to add this one or not , ;_;
        spec_PCEN = librosa.pcen(spec* (2**31), sr=fs/10).astype(np.float32)

        means = np.mean(spec, axis = 1)
        meansECPN = np.mean(spec_PCEN, axis = 1)

        for i in range(len(means)):
            spec[i] = np.maximum(spec[i] - means[i] , 0)
            spec_PCEN[i] = np.maximum(spec_PCEN[i] - meansECPN[i], 0)

        for index, row in df[df['id'] == filename].iterrows():
            startNum = int(row['trimInTime'] / fileDuration * spec.shape[1]) - 4 
            endNum   = int(row['trimOutTime'] / fileDuration * spec.shape[1]) -4
            trimmed_spec = spec[0:spec.shape[0],startNum:endNum]
            trimmed_spec_PCEN = spec_PCEN[0:spec.shape[0],startNum:endNum]
            if(trimmed_spec.shape[1] == 33):
                trimmed_spec = trimmed_spec[0:trimmed_spec.shape[0],0:32]
                trimmed_spec_PCEN = trimmed_spec_PCEN[0:trimmed_spec.shape[0],0:32]
    #         if((trimmed_spec.shape[1] != 33) and (trimmed_spec.shape[1] != 32)):
    #             continue
            trimmed_data.append(trimmed_spec)
            trimmed_data_PCEN.append(trimmed_spec_PCEN)

    return trimmed_data, trimmed_data_PCEN

def addNumLabels(df):

    labels = []
    for index, row in df.iterrows():
        label = classesDic[row['class']]
        labels.append(label)
    
    return labels

def reSizeSpec(df):
    data = df['data'].tolist()
    normData = df['normData'].tolist()
    reShapedSpecs = []
    reShapedSpecsPCEN = []
    for spec in data:
        reSpec = cv2.resize(spec, (32,128))
        reShapedSpecs.append(reSpec)
        
    for spec in normData:
        reSpec = cv2.resize(spec, (32,128))
        reShapedSpecsPCEN.append(reSpec)

    # print(reShapedSpecs)

    # reShapedSpecs = np.array(reShapedSpecs)

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=0, std=1)
    # ])

    # reShapedSpecs = transform(reShapedSpecs).tolist()

    return reShapedSpecs, reShapedSpecsPCEN
    


def getDataLoader(batchSize,ifAug,ifTrain):

    noiseList = []
    dfList = []

    if ifTrain:
        audioDic = trainAudioDic
    else:
        audioDic = testAudioDic

    for annName in audioDic:
        annPath = 'annotations/' + annName
        audioPath = 'audio/' + audioDic[annName]
        df = dfLoader(annPath, audioPath, allData = True)
        noises, noise_PCEN = getNoiseList(df, audioPath, 300)
        noiseList.extend(noises)
        dfList.append(df)

    finalDF = pd.concat(dfList)
    # print(finalDF['class'].unique())
    dataLoader = genDataLoader(batchSize,finalDF,ifAug, noiseList)
    return dataLoader

def getDataLoaderSplit(batchSize,ifAug,ifTrain, splitPercent):

    noiseList = []
    dfList = []


    if ifTrain:
        audioDic = trainAudioDic
        noiseAmount = 300
    else:
        audioDic = testAudioDic
        noiseAmount = 0

    for annName in audioDic:
        annPath = 'annotations/' + annName
        audioPath = 'audio/' + audioDic[annName]
        df = dfLoader(annPath, audioPath, allData = False)
        noises, noise_PCEN = getNoiseList(df, audioPath, noiseAmount)
        noiseList.extend(noises)
        dfList.append(df)

    finalDF = pd.concat(dfList)
    # print(finalDF['class'].unique())
    dataLoader = genDataLoaderSplit(batchSize,finalDF,ifAug, noiseList, splitPercent)
    return dataLoader

def getNoiseList(df, audioPath, numNoise):

    if numNoise == 0:
        return []

    noiseDic = {}
    for name, group in df.groupby('id'):
        noisePeriod = []
        startTime = group['start_time'].tolist()
        endTime = group['end_time'].tolist()
        duration = group['duration'].tolist()[0]
        for i in range(len(endTime)):
            if i == 0:
                noisePeriod.append((0,startTime[0]))
            elif i == (len(endTime) - 1):
                noisePeriod.append((endTime[i], duration))
            else:
                noisePeriod.append((endTime[i-1], startTime[i]))
        noiseDic[name] = noisePeriod


    trimmed_noise = []
    trimmed_noise_PCEN = []
    for filename in df['id'].unique():
        fs, x = wavfile.read(audioPath + filename)
        fileDuration = x.shape[0]/fs
        spec = gen_mag_spectrogram(x, 1024, 840)

        unit = fs/2/spec.shape[0]
        startP = int(10000/unit)
        endP = int(140000/unit) + 1

        spec = spec[startP:endP]

        specPCEN = librosa.pcen(spec* (2**31), sr=fs/10).astype(np.float32)

        means = np.mean(spec, axis = 1)
        meansECPN = np.mean(specPCEN, axis = 1)
        
        for i in range(len(means)):
            spec[i] = np.maximum(spec[i] - means[i] , 0)
            specPCEN[i] = np.maximum(specPCEN[i] - meansECPN[i], 0)


        for (startTime, endTime) in noiseDic[filename]:
            if(endTime - startTime >= 0.02):
                endNum = int(endTime  / fileDuration * spec.shape[1])-6
                startNum = endNum - 32
                trimmed_spec = spec[0:spec.shape[0],startNum:endNum]
                trimmed_spec_PCEN = specPCEN[0:spec.shape[0],startNum:endNum]
                if(trimmed_spec.shape[1] != 32):
                    continue
                reshaped_spec = cv2.resize(trimmed_spec,(32,128))
                reshaped_spec_PCEN = cv2.resize(trimmed_spec_PCEN, (32,128))
                trimmed_noise.append(reshaped_spec)
                trimmed_noise_PCEN.append(reshaped_spec_PCEN)

    trimmed_noise, trimmed_noise_PCEN = zip(*random.sample(list(zip(trimmed_noise, trimmed_noise_PCEN)), numNoise))

    return trimmed_noise, trimmed_noise_PCEN
    

def genDataLoaderFromDataset(datasetPath, dataAug, augAmount, batchSize):
    dataset = torch.load(datasetPath)
    X = dataset.tensors[0]
    y = dataset.tensors[1]
    
    if dataAug:
        X_aug = genAugData(torch.clone(X), augAmount)
        X = torch.cat((X,X_aug))
        y = y.repeat(augAmount + 1)
        print(X.shape)
        print(y.shape)
    datasetTensor = TensorDataset(X,y)
    dataLoader = DataLoader(datasetTensor, batch_size = batchSize, shuffle=True)
    
    return dataLoader

def genAllDataFrame(audioDic):
    dfList = []

    for annName in audioDic:
        annPath = 'annotations/' + annName
        audioPath = 'audio/' + audioDic[annName]
        df = dfLoader(annPath, audioPath,False)
        dfList.append(df)

    all_DF = pd.concat(dfList)

    return all_DF

def genNoiseList(noiseCount):

    noiseList = []
    noiseListPCEN = []

    for annName in annAudioDic:
        annPath = 'annotations/' + annName
        audioPath = 'audio/' + annAudioDic[annName]
        df = dfLoader(annPath, audioPath, allData = False)
        noises, noise_PCEN = getNoiseList(df, audioPath, int(noiseCount/8))
        noiseList.extend(noises)
        noiseListPCEN.extend(noise_PCEN)

    return noiseList, noiseListPCEN

def genTrainTestLoader(batchSize, dataAug, dataCount, testAmount):

    
    all_DF = genAllDataFrame(annAudioDic)

    classNumDic = {}
    for index, value in all_DF['class'].value_counts().items():
        classNumDic[index] = value * (1-testAmount)

    classNumDicDS = {}

    for i in classes:
        classNumDicDS[i] = dataCount
        
    classDataDic = {}

    X_train = []
    y_train = []

    X_test = []
    y_test = []
    
    
    
    files = all_DF['id'].unique()
    random.shuffle(files)

    for file in files:
        curr_df = all_DF[(all_DF['id'] == file)]

        for index, row in curr_df.iterrows(): 
            
            rowClass = row['class']
            
            if(classNumDic[rowClass] > 0):
                
                if rowClass in classDataDic:
                    
                    classDataDic[rowClass].append(row['data'])
                else:
                    classDataDic[rowClass] = [row['data']]

#                 if(classNumDicDS[rowClass] > 0):

#                     X_train.append(row['data'])
#                     y_train.append(row['label'])
#                     classNumDicDS[rowClass] = classNumDicDS[rowClass]-1

                classNumDic[rowClass] = classNumDic[rowClass]-1
            else:
                X_test.append(row['data'])
                y_test.append(row['label'])
                
    for key in classDataDic:
        sampled_class_data = random.sample(classDataDic[key], dataCount)
        X_train.extend(sampled_class_data)
        y_train.extend([classesDic[key]]*dataCount)
        

    noiseList = genNoiseList(dataCount)
    X_train.extend(noiseList)
    y_train.extend([17]*len(noiseList))

    X_train = torch.Tensor(np.array(X_train))
    y_train = torch.Tensor(np.array(y_train))

    X_test = torch.Tensor(np.array(X_test))
    y_test = torch.Tensor(np.array(y_test))

#     print(len(X_train))
#     print(len(y_train))

#     for i in range(17):
#         print(f'number of sample of class {i} is {y_test.tolist().count(i)}')

#     for i in range(17):
#         print(f'number of sample of class {i} is {y_train.tolist().count(i)}')

    if dataAug:
        augData = genAugData(X_train,3)
        print(f'Data Aug size is {augData.size()}')

        X_train = torch.concat((X_train,augData))
        y_train = torch.concat((y_train,y_train,y_train,y_train))

    datasetTensor = TensorDataset(X_train,y_train)
    trainLoader = DataLoader(datasetTensor, batch_size = batchSize, shuffle=True)

    datasetTensor = TensorDataset(X_test,y_test)
    testLoader = DataLoader(datasetTensor, batch_size = batchSize, shuffle=True)
    
#     print(train_files)
    
    return trainLoader, testLoader



# trainLoader, testLoader = genTrainTestLoader(batchSize = 32, dataAug = True, dataCount = 302, testAmount = 0.25)
