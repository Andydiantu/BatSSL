from dfLoader import *
import torch
import sklearn.model_selection as ms
from collections import Counter



def SaveDatasets(df, noiseList):

    X = []
    y = []

    for index, row in df.iterrows():
        reshapeData = row['data']
        X.append(reshapeData)
        y.append(row['label'])

    X.extend(noiseList)
    y.extend([17]*len(noiseList))

    X = torch.Tensor(np.array(X))
    y = torch.Tensor(np.array(y))

    
    print(f'data size for bat is {X.size()}')
    print(f'size for label is {y.size()}')

    print()
    
    split_sizes = [0.0,0.5,0.8,0.5,0.8]
    split_protion = [100,50,10,5,1]
    
    splits = zip(split_sizes,split_protion)
    
    for split_size, split_protion in splits:
        if(split_size != 0.0):
            X, _, y, _ = ms.train_test_split(X, y, test_size=split_size)
        datasetTensor = TensorDataset(X,y)
        print(f'dataset size is {len(datasetTensor)}')
        torch.save(datasetTensor, 'datasets/dataSet_'+str(split_protion)+'.pt')

    
    

#     dataLoader = DataLoader(datasetTensor, batch_size = batchSize, shuffle=True)

    return datasetTensor


def genDatasets(ifTrain):

    noiseList = []
    dfList = []

    if ifTrain:
        audioDic = trainAudioDic
    else:
        audioDic = testAudioDic

    for annName in audioDic:
        annPath = '../annotations/' + annName
        audioPath = '../audio/' + audioDic[annName]
        df = dfLoader(annPath, audioPath, allData = False)
        noises = getNoiseList(df, audioPath, 300)
        noiseList.extend(noises)
        dfList.append(df)

    finalDF = pd.concat(dfList)
    # print(finalDF['class'].unique())
    dataLoader = SaveDatasets(finalDF, noiseList)
    return dataLoader


def genDatasetBalanced(testAmount, dataCount):

    all_DF = genAllDataFrame(annAudioDic)

    classNumDic = {}
    for index, value in all_DF['class'].value_counts().items():
        classNumDic[index] = value * (1-testAmount)
        
    classDataDic = {}
    classDataPCENDic = {}

    

    X_test = []
    X_test_PCEN = []
    y_test = []
    y_test_PCEN = []

    X_train_unlabbel = []
    X_train_unlabbel_PCEN = []

    train_files = set()
    test_files = set()


    files = all_DF['id'].unique()
    random.shuffle(files)

    for file in files:
        curr_df = all_DF[(all_DF['id'] == file)]

        for index, row in curr_df.iterrows(): 
            rowClass = row['class']
            
            if(classNumDic[rowClass] > 0):
                
                if(file in test_files):
                    continue

                if rowClass in classDataDic:
                    classDataDic[rowClass].append(row['data'])
                    classDataPCENDic[rowClass].append(row['normData'])
                else:
                    classDataDic[rowClass] = [row['data']]
                    classDataPCENDic[rowClass] = [row['normData']]
                classNumDic[rowClass] = classNumDic[rowClass]-1

                train_files.add(file)
            else:
                if file in train_files:
                    continue
                X_test.append(row['data'])
                y_test.append(row['label'])

                X_test_PCEN.append(row['normData'])
                y_test_PCEN.append(row['label'])

                test_files.add(file)

    print(train_files.intersection(test_files))
    print(test_files.intersection(train_files))

    print(f'size of testing data is {len(X_test)}')
    print(f'size for unlabbled data is {len(X_train_unlabbel)}')

    noiseList, noise_PCEN = genNoiseList(400)

    classDataDic['noise'] = noiseList
    classDataPCENDic['noise'] = noise_PCEN

    for key in classDataDic:
            X_train_unlabbel.extend(classDataDic[key])
            X_train_unlabbel_PCEN.extend(classDataPCENDic[key])
    
    split_protion = [300,150,30,15,3]

    for dataNum in split_protion:
        
        X_train = []
        X_trian_PCEN = []
        y_train = []
        y_train_PCEN = []

        for key in classDataDic:
            

            # print(len(classDataDic[key]))
            # print(len(classDataPCENDic[key]))
            # print(dataNum)
            # print(key)
            # print()

            zippedList = list(zip(classDataDic[key], classDataPCENDic[key]))
            sampled_class_data, sampled_class_data_PCEN = zip(*random.sample(zippedList, dataNum))
            
            X_train.extend(sampled_class_data)
            y_train.extend([classesDic[key]]*dataNum)
            X_trian_PCEN.extend(sampled_class_data_PCEN)
            y_train_PCEN.extend([classesDic[key]]*dataNum)

            classDataDic[key] = sampled_class_data
            classDataPCENDic[key] = sampled_class_data_PCEN
        print(f'size of training data is {len(X_train)}')
        SaveDatasets(f'dataset_{dataNum}', X_train, y_train)
        SaveDatasets(f'dataset_PCEN_{dataNum}', X_trian_PCEN, y_train_PCEN)


    SaveDatasets('Test_dataSet', X_test, y_test)
    SaveDatasets('Test_dataSet_PCEN', X_test_PCEN, y_test_PCEN)

    
   

    nullLabels = [0] * len(X_train_unlabbel)
    SaveDatasets('Unlabbeled_train_dataSet', X_train_unlabbel, nullLabels)
    SaveDatasets('Unlabbeled_train_dataSet_PCEN', X_train_unlabbel_PCEN, nullLabels)
    

def SaveDatasets(name, X, y):

    unique_items, unique_counts = zip(*Counter(y).items())

    print(f'its {name}')
    print("Unique items:", unique_items)
    print("Counts of unique items:", unique_counts)

    X = torch.Tensor(np.array(X))
    y = torch.Tensor(np.array(y))

    datasetTensor = TensorDataset(X,y)
    print(f'dataset {name} size is {len(datasetTensor)}')
    torch.save(datasetTensor, f'src/datasets/{name}.pt')

    print()

# def genDatasetBalanced(testAmount, dataCount):

#     all_DF = genAllDataFrame(annAudioDic)

#     classNumDic = {}
#     for index, value in all_DF['class'].value_counts().items():
#         classNumDic[index] = value * (1-testAmount)
        
#     classDataDic = {}

#     X_train = []
#     y_train = []

#     X_test = []
#     y_test = []

#     X_train_unlabbel = []

#     train_files = set()
#     test_files = set()


#     files = all_DF['id'].unique()
#     random.shuffle(files)

#     for file in files:
#         curr_df = all_DF[(all_DF['id'] == file)]

#         for index, row in curr_df.iterrows(): 
#             rowClass = row['class']
            
#             if(classNumDic[rowClass] > 0):
                
#                 if(file in test_files):
#                     continue

#                 if rowClass in classDataDic:
#                     classDataDic[rowClass].append(row['data'])
#                 else:
#                     classDataDic[rowClass] = [row['data']]

#                 classNumDic[rowClass] = classNumDic[rowClass]-1

#                 train_files.add(file)
#             else:
#                 if file in train_files:
#                     continue
#                 X_test.append(row['data'])
#                 y_test.append(row['label'])

#                 test_files.add(file)

#     print(train_files.intersection(test_files))
#     print(test_files.intersection(train_files))


    


#     for key in classDataDic:
#         X_train_unlabbel.extend(classDataDic[key])
#         sampled_class_data = random.sample(classDataDic[key], dataCount)
#         X_train.extend(sampled_class_data)
#         y_train.extend([classesDic[key]]*dataCount)
        
    
#     noiseList, noise_PCEN = genNoiseList(dataCount)
#     X_train.extend(noiseList)
#     y_train.extend([17]*len(noiseList))

#     X_train = torch.Tensor(np.array(X_train))
#     y_train = torch.Tensor(np.array(y_train))

#     X_test = torch.Tensor(np.array(X_test))
#     y_test = torch.Tensor(np.array(y_test))

#     nullLabels = [0] * len(X_train_unlabbel)
#     X_train_unlabbel = torch.Tensor(np.array(X_train_unlabbel))
#     nullLabels = torch.as_tensor(nullLabels)

#     # save datasets as .pt file

#     print(f'data size for bat is {X_train.size()}')
#     print(f'size for label is {y_train.size()}')

#     print()
    
#     split_sizes = [0.0,0.5,0.8,0.5,0.8]
#     split_protion = [100,50,10,5,1]
    
#     splits = zip(split_sizes,split_protion)
    
#     for split_size, split_protion in splits:
#         if(split_size != 0.0):
#             X_train, _, y_train, _ = ms.train_test_split(X_train, y_train, test_size=split_size)
#         datasetTensor = TensorDataset(X_train,y_train)
#         print(f'dataset size is {len(datasetTensor)}')
#         torch.save(datasetTensor, 'src/datasets/dataSet_'+str(split_protion)+'.pt')

#     datasetTensor = TensorDataset(X_test, y_test)
#     print(f'dataset size is {len(datasetTensor)}')
#     torch.save(datasetTensor, 'src/datasets/Test_dataSet.pt')

    
#     datasetTensor = TensorDataset(X_train_unlabbel, nullLabels)
#     print(f'dataset size is {len(datasetTensor)}')
#     torch.save(datasetTensor, 'src/datasets/Unlabbeld_train_dataSet.pt')




############################ Execution Code ############################


# genDatasets(True)
genDatasetBalanced(0.2, 302)