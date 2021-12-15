#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import os
import json
import argparse
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F


# mostly used the course materials and pytorch website and for the code mostly used the following aws repo
# https://github.com/aws/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/pytorch_mnist/hpo_pytorch_mnist.ipynb

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def test(model, criterion, test_loader, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("Started testing the model")
    model.eval()
    running_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            running_loss += loss.item() * data.size(0)

    test_loss = running_loss / len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, criterion, optimizer, device, args):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("Started tarining the model")
    for epoch in range(args.epochs):
        model.train()
        running_loss=0
        correct=0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(data)             #No need to reshape data since CNNs take image inputs
            loss = criterion(pred, target)
            running_loss+=loss
            loss.backward()
            optimizer.step()
            pred=pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        logger.info(
            f"""Epoch {epoch}: 
                Loss {running_loss/len(train_loader.dataset)}, 
                Accuracy {100*(correct/len(train_loader.dataset))}%
             """
        )
    return model
    
def net():
    '''
    summary: initiazlizes a pretrained model and sets its parameters

    returns:
        the initialized model
    '''
    model = models.resnet18(pretrained=True)

    # freeze the CNN layers
    for param in model.parameters():
        param.require_grad = False
    
    num_features=model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 133))

    return model

def get_data_loader(dir_path, batch_size, mode):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    logger.info(f"Get data for mode = {mode} from {dir_path}")
    
    transformations = {
        "train": transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        ,
        "test": transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    }

    dataset = ImageFolder(dir_path, transform=transformations[mode])

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )


def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    # check if GPU is available use it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on Device {device}")
    model=net()
    model=model.to(device)  

    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader = get_data_loader(args.train_data_dir, args.batch_size, "train")
    test_loader = get_data_loader(args.test_data_dir, args.test_batch_size, "test")

    model=train(model, train_loader, loss_criterion, optimizer, device, args)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, loss_criterion, test_loader, device)
    
    '''
    TODO: Save the trained model
    '''
    
    path = os.path.join(args.model_dir, "model.pth")
    logger.info(f"Saving the model to {path}")
    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    
    # hyperparameters and data parsers
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        metavar="N",
        help="number of epochs to train (default: 4)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train-data-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test-data-dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--val-data-dir", type=str, default=os.environ["SM_CHANNEL_VAL"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    
    args=parser.parse_args()
    
    main(args)
