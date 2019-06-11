from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random as random
import numpy as np
import scipy.optimize
from torchvision import datasets, transforms, models
import sys
import pickle
import os
import copy
from model import *

def train(args, model, device, train_loader, integral_loader, optimizer, epoch):
    model.train()
    for batch_idx, (datax, targetx) in enumerate(train_loader):
        datax, targetx = datax.to(device), targetx.to(device)
        datay, targety = next(iter(integral_loader))
        datay, targety = datay.to(device), targety.to(device)

        optimizer.zero_grad()

        outx = model(datax)
        outy = model(datay)
        loss1 =  torch.mean(torch.log(outx)) 
        loss2 = torch.log(torch.mean(outy))

        loss = -loss1 + loss2
        loss.backward()
        optimizer.step()

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(datax), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
    return(loss.item())


def average_weights(model, device, integral_loader):
    model.eval()
    estim_coins = np.zeros(2)
    counts = np.zeros(2)
    with torch.no_grad():
        for data, target in integral_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            for i, x in enumerate(target):
                estim_coins[target[i].item()] += output[i].item()
                counts[target[i].item()] += 1

        estim_coins=estim_coins/counts
    return estim_coins

def countfiles(dir): 
    return len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=400, metavar='N',
                        help='input batch size for testing (default: 400)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-to', 
                        help='for Saving LOGS and Model')
    parser.add_argument('--data', 
                        help='data directory')
    parser.add_argument('--continue-from', 
                        help='give NN to continue training directory')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    image_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    imagedir = args.data
    if imagedir[-1]=="/":
        imagedir=imagedir[:-1]

    train_data = datasets.ImageFolder(root=imagedir+"/train", transform=image_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)

    test_data = datasets.ImageFolder(root=imagedir+"/test", transform=image_transform)
    test_loader= torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    out_string = os.path.basename(imagedir)+"_lr="+ str(args.lr) +"_E="+str(args.epochs)
    sys.stdout = open(args.save_to+"/LOG_GEN_WEIGHT_"+ out_string +".txt", 'w')
 
    model = AlexWeightNN()
    if args.continue_from is not None:
        model.load_state_dict(torch.load(args.continue_from))

    model = model.to(device)
    best_nn = copy.deepcopy(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        new_loss = train(args, model, device, train_loader, test_loader, optimizer, epoch)
        outp = average_weights(model, device, train_loader)
        if new_loss < loss:
            loss = new_loss
            print("Change NN")
            best_nn = copy.deepcopy(model)
        print(outp)

    torch.save(best_nn.state_dict(),args.save_to+"/NN_GEN_WEIGHT_"+ out_string +".pt")

if __name__ == '__main__':
    main()
