import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
from utils import dataLoader
from tempfile import TemporaryDirectory

def instantiateModel(weights, numClasses, learningRate, momentum):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=weights)
    numFeatures = model.fc.in_features
    model.fc = nn.Linear(numFeatures, numClasses)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=momentum)
    lrScheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return model, criterion, optimizer, lrScheduler

def train(model, saveDir, dataLoader, datasetSizes, criterion, optimizer, scheduler, numEpochs):
    since = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best_model_params_path = os.path.join(saveDir, 'best_model_params.pt')

    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    for epoch in range(numEpochs):
        print(f'Epoch {epoch}/{numEpochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataLoader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / datasetSizes[phase]
            epoch_acc = running_corrects.double() / datasetSizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')



    # save & load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    torch.save(model.state_dict(), os.path.join(saveDir, f'best_model_{numEpochs}.pt'))
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--numclasses", type=int, default=3)
    parser.add_argument("--savedir", type=str, default="/content/weights", help="Directory of weights")
    parser.add_argument("--datadir", type=str, help="Directory of dataset")
    parser.add_argument("--batchsize", type=str, default=4, help="Number of batches")
    parser.add_argument('--device', type=str, default="cpu", help="Device Type for Training")
    parser.add_argument("--weights", type=str, default="IMAGENET1K_V1")
    parser.add_argument("--learningrate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    opt = parser.parse_args()

    dataset, dataLoader, datasetSizes = dataLoader(opt.datadir, opt.batchsize)
    nnModel, criterion, optimizer, lrScheduler= instantiateModel(opt.weights, opt.numclasses, opt.learningrate, opt.momentum)
    model = train(nnModel, opt.savedir, dataLoader, datasetSizes, criterion, optimizer, lrScheduler, opt.epochs)


