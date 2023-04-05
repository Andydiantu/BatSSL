from dfLoader import *
from dataAug import *
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mat
mat.rcParams.update({'figure.max_open_warning': 0})

def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(spec, origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    fig.set_size_inches(3, 10)
    plt.savefig('images/' + title)
    # plt.show(block=False)

def getDataLoader():

    dfList = []
    
    audioDic = annAudioDic

    for annName in audioDic:
        annPath = '../annotations/' + annName
        audioPath = '../audio/' + audioDic[annName]
        df = dfLoader(annPath, audioPath, False)
        dfList.append(df)

    finalDF = pd.concat(dfList)
    return finalDF


classes = ['Myotis daubentonii', 'Plecotus auritus', 'Pipistrellus pipistrellus' ,'Bat',
 'Nyctalus leisleri' ,'Pipistrellus pygmaeus', 'Myotis mystacinus',
 'Myotis nattereri', 'Pipistrellus nathusii', 'Nyctalus noctula',
 'Eptesicus serotinus', 'Barbastellus barbastellus', 'Myotis brandtii',
 'Myotis alcathoe' ,'Myotis bechsteinii', 'Plecotus austriacus',
 'Rhinolophus ferrumequinum', 'Rhinolophus hipposideros']


all_DF = getDataLoader()

for className in classes:
    data = np.array(all_DF[all_DF['class'] == className].data.tolist())
    # test_data = np.array(test_DF[test_DF['class'] == className].data.tolist())
    
    
    if(data.size != 0):
        avg = np.average(data,axis=0)
        avg_fileName = 'train ' + className + '.pdf'
        plot_spectrogram(np.flip(avg,0),avg_fileName)




