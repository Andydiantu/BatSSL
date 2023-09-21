# Bat SSL User Guide

## Getting started

1) Install the miniconda for your operating system from [here](https://docs.conda.io/en/latest/miniconda.html).
2) Download this code from the repository (by clicking on the green button on top right) and unzip it. 
3) Create a new environment and install the required packages:
`conda env create -f environment.yml`
`conda activate SSLBAT`.

## Code Explaination

### Audio Preprocessing

`src/dfLoader.py` contains function involves loading raw audio file and annotation files, converting audio files to spectrogram, and trimming out spectrograms which only contrain one call according to the annotation given. 

`src/dataAug.py` contains function involves all data augmentation applied, and function to apply all augmentations to each spectrogram in a tensor. 

`src/genDatasets.py` contains function to generate static dataset files, including split train test dataset and generate nested subsampled train dataset. 

### Baseline Code

`src/resNet18.py` contains code of implementing a resNet-18 structure

`src/batNet.py` code to train the baseline model from scratch.

`src/resNet18LinEval.py` linear evaluate the supervise trained baseline model.

### Autoencoder Code

`src/autoEncoder.py` contains code of implementing a resnet-18 based autoencoder.

`src/autoEncoderTrain.py` pretrain the autoencoder in self-supervised manner from scratch.

`src/autoEncoderFinetune.py` finetune the pretrained autoencoder model.


### SimCLR code

`src/SimCLR.py` contains code of implementing a resnet-18 base SimCLR model.

`src/SimCLRTrain.py` pretrain the SimCLR model in self-supervised manner from scratch.

`src/SimCLRFinetune.py` finetune the pretrained self-supervised model.

### Other code

`src/evalModel.py` evaluate the model performance. 

`src/logisticRegression.py` linear evaluate the pretrained model using mutinomial logistic regression. 

`src/pltClassImg.py` plot the average image for each class.


## Running the model on your own data

After you follow the steps in the *Getting started* section, you could try train the model using you own data by running the following command.

### Self-supervised pretraining

**Autoencoder**: `python src/autoEncoderTrain.py --epochs_train NUMOFTRAININGEPOCH --modelOutputName MODELOUTPUTNAME`

**SimCLR**: `python src/contrastNet.py --epochs_train NUMOFTRAININGEPOCH --modelOutputName MODELOUTPUTNAME`

### Finetuning

**Autoencoder**: `python src/autoEncoderFinetune.py --epochs_Finetune NUMOFTRAININGEPOCH --linEval False --datasetPath PATHTODATASET --modelOutputName PATHTOPRETRAINEDMODEL`

**SimCLR**: `python src/SimCLRFinetune.py --epochs_Finetune NUMOFTRAININGEPOCH --linEval False --datasetPath PATHTODATASET --modelOutputName PATHTOPRETRAINEDMODEL`
