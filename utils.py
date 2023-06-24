import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def showImages(inputs, objClasses, class_names):
    count = 0
    for input, objClass in zip(inputs, objClasses):
        count += 1
        input = input.numpy().transpose((1, 2, 0))  # Convert to 2D array
        mean = np.array([0.485, 0.456, 0.406]) #Mean from ImageNet
        std = np.array([0.229, 0.224, 0.225]) #Std Dev from ImageNet
        input = std * input + mean
        input = np.clip(input, 0, 1)

        print(count)
        plt.subplot(int(len(inputs) / 2) + 1, 2, count)
        plt.imshow(input)
        plt.title(class_names[objClass])

        plt.tight_layout(pad=0.5)

    return

def showImage(input, title):
    input = input.numpy().transpose((1, 2, 0))  # Convert to 2D array
    mean = np.array([0.485, 0.456, 0.406]) #Mean from ImageNet
    std = np.array([0.229, 0.224, 0.225]) #Std Dev from ImageNet
    input = std * input + mean
    input = np.clip(input, 0, 1)

    plt.imshow(input)
    plt.title(title)


def transformConfig():
    transformConfig = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # Use mean & std dev from ImageNet (Official PyTorch Implementation)
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # Use mean & std dev from ImageNet (Official PyTorch Implementation)
        ]),
        'test': transforms.Compose([
            transforms.Resize(300),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # Use mean & std dev from ImageNet (Official PyTorch Implementation)
        ]),
    }
    return transformConfig

def dataLoader(dataDir, batchSize):
    transform = transformConfig()
    dataset = {x: datasets.ImageFolder(os.path.join(dataDir, x),transform[x]) for x in ['train', 'val', 'test']}
    dataLoader = {x: torch.utils.data.DataLoader(dataset[x], batch_size = batchSize, shuffle=True, num_workers=4)
                  for x in ['train', 'val', 'test']}
    datasetSizes = {x: len(dataset[x]) for x in ['train', 'val', 'test']}
    class_names = dataset['train'].classes
    print(f"Classes:{class_names}")

    return dataset, dataLoader, datasetSizes

if __name__ == "__main__":
    pass