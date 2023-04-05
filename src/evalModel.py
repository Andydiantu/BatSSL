import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import torch.nn as nn
import sklearn.metrics.pairwise as pw
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def save_plots(train_acc, train_loss, valid_acc, valid_loss, fileName):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'outputs/{fileName}_accuracy.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'outputs/{fileName}_loss.png')

def evalModelNew(model, testLoader, classNames):
    trueLabels = []
    predictLabels = []

    with torch.no_grad():
        correct = 0
        total = 0
        n_class_correct = [0 for i in range(len(classNames))]
        n_class_samples = [0 for i in range(len(classNames))]
        for images, labels in testLoader:
            images = images.to(device)
            labels = labels.to(device).type(torch.int64)
            if(len(labels) < 32):
                break
            # calculate outputs by running images through the network
            outputs = model(images.view(-1,1, 128, 32))
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            trueLabels.extend(labels.tolist())
            predictLabels.extend(predicted.tolist())
            correct += (predicted == labels).sum().item()
            
            for i in range(32):
                label = labels[i]
                pred = predicted[i]

                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1
                
        acc = 100.0 * correct / total
        print(f'Accuracy of the network: {acc} %')
        print()
        print()
        print(classification_report(trueLabels,predictLabels))
        print()
        print()
        
        balanced_acc = 0.0

        for i in range(len(classNames)):
            if(n_class_samples[i] == 0):
                continue
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]

            balanced_acc += (acc / 17)
            print(f'Accuracy of {classNames[i]}: {acc} %')
        
        print(f'The balanced accuracy is {balanced_acc}')

        cm = confusion_matrix(trueLabels, predictLabels, normalize='true') 
        

        desired_order = [10, 9, 12, 13, 11, 0, 5, 6, 3, 8, 7, 2, 4, 1, 14, 15, 16, 17]

        fig, ax = plt.subplots(figsize=(20, 20))

        # Plot the confusion matrix without color bar

        cm = cm[desired_order, :]
        cm = cm[:, desired_order]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(classNames))

        cax = disp.plot(xticks_rotation=50, ax=ax, cmap='viridis', colorbar=False)


        # Customize the text to display at most 2 decimal places
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                disp.text_[i, j].set_text(f'{cm[i, j]:.2f}')
                
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
        ax.set_xlabel('Predicted label', fontsize=14)
        ax.set_ylabel('True label', fontsize=14)

        # Add the color bar at the bottom
        cbar = plt.colorbar(cax.im_, ax=ax, shrink=0.6, orientation='horizontal')
        cbar.ax.tick_params(labelsize=10)
        plt.tight_layout()
        fig.savefig('confu_baseline.png', dpi = 300)

        # cm = cm.round(3)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm,
        #                        display_labels=classNames)


        # fig, ax = plt.subplots(figsize=(20,20))

        # f = disp.plot(xticks_rotation = 50, ax = ax)
        # fig.savefig('confu_simCLR.png', dpi = 300)

        return trueLabels, predictLabels


def evalModelPerEpoch(model, validLoader, classNames, criterion):
    trueLabels = []
    predictLabels = []
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        correct = 0
        total = 0
        losses = []
        n_class_correct = [0 for i in range(len(classNames))]
        n_class_samples = [0 for i in range(len(classNames))]
        for images, labels in validLoader:
            images = images.to(device)
            labels = labels.to(device).type(torch.int64)
            if(len(labels) < 32):
                break
            # calculate outputs by running images through the network
            outputs = model(images.view(-1,1, 128, 32))

            loss = criterion(outputs, labels)
            losses.append(loss.item())
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            trueLabels.extend(labels.tolist())
            predictLabels.extend(predicted.tolist())
            correct += (predicted == labels).sum().item()
            
            for i in range(32):
                label = labels[i]
                pred = predicted[i]

                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1
        mean_loss = sum(losses)/len(losses)
        acc = 100.0 * correct / total
    return acc, mean_loss


def contrastAccEval(embeddings, indices):
    posiCount = 0
    embeddings = torch.clone(embeddings).cpu().detach()
    exp = torch.from_numpy(pw.cosine_similarity(embeddings,embeddings))
    for i in range(embeddings.size(dim=0)):
        if(i >= embeddings.size(dim=0)/2):
            break

        maxIndex = torch.topk(exp[i], 2).indices[1]
        if (indices[i] == indices[maxIndex]):
            posiCount += 1
            # print(j)
            # print(i)
                            
    # print(posiCount)
    # print(embeddings.size(dim=0))
    # return 100 * (posiCount/(embeddings.size(dim=0)/2))
    
    return posiCount


def plot_encoder_result(spec, spec2, title, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    axs[0].set_title( "Origin")
    axs[0].set_ylabel(ylabel)
    axs[0].set_xlabel("frame")
    im0 = axs[0].imshow(spec, origin="lower", aspect=aspect)
    
    axs[1].set_title("Reconstructed")
    axs[1].set_xlabel("frame")
    im1 = axs[1].imshow(spec2, origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im0, ax=axs[0])
    fig.colorbar(im1, ax=axs[1])
    fig.set_size_inches(8, 10)
    fig.suptitle(title, fontsize=16)

    
    plt.savefig(f'outputs/{title}.png', dpi=150)


def printClassAccuracy(classNames, trueLabels, predictedLabels):
    n_class_correct = [0 for i in range(len(classNames))]
    n_class_samples = [0 for i in range(len(classNames))]
    
    for i in range(len(trueLabels)):
        label = int(trueLabels[i])
        pred  = predictedLabels[i]

        if (label == pred):
            n_class_correct[label] += 1
        n_class_samples[label] += 1

    balanced_acc = 0.0

    for i in range(len(classNames)):
        if(n_class_samples[i] == 0):
            continue
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]

        balanced_acc += (acc / 17)
        print(f'Accuracy of {classNames[i]}: {acc} %')
    
    print(f'The balanced average accuracy is {balanced_acc}')

    
