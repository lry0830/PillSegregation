import argparse
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from utils import transformConfig, dataLoader, showImage

def predict(modelPath, dataLoader, device, numImages, classes, classNames):

    imgCount = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    numFeatures = model.fc.in_features
    model.fc = nn.Linear(numFeatures, len(classNames))
    model.load_state_dict(torch.load(modelPath))
    model.to(device)
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataLoader['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                imgCount += 1
                showImage(inputs.cpu().data[j], f'predicted: {classNames[preds[j]]}')

                #plt.ioff()
                plt.show()

                if imgCount == numImages:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0", help="Device Type for Training")
    parser.add_argument("--weights", type=str, default="/content/weight/best.pt")
    parser.add_argument("--datadir", type=str, help="Directory of dataset")
    parser.add_argument("--numimages", type=int, help="Number of images")
    opt = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset, dataLoader2, _ = dataLoader(opt.datadir, 1)
    inputs, classes = next(iter(dataLoader2['train']))
    classNames = dataset['test'].classes
    predict(opt.weights, dataLoader2, device, opt.numimages, classes, classNames)



