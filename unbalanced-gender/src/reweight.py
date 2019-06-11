import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import random as random
import sys
import numpy as np
import os
import copy
from model import *

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    lossf = nn.CrossEntropyLoss(reduction='mean')
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossf(output, target)
        loss.backward()
        optimizer.step()

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
           epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, num_classes):
    model.eval()
    test_loss = 0
    correct = np.zeros(num_classes)
    counts = np.zeros(num_classes)

    lossf = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = lossf(output, target)
            test_loss += loss
            pred = output.argmax(dim=1, keepdim=True)
            for i, x in enumerate(target):
                counts[x] += 1.
                if (x == pred[i]):
                    correct[x] += 1.  
    

    test_loss /= len(test_loader.dataset)
    per_class_acc = correct/counts
    total_acc =  sum(correct)/len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}'.format(test_loss))
    print('Test set: Total Accuracy: {:.2f}%'.format(100. * total_acc))
    print('Test set: Accuracy Per Class: {}'.format(per_class_acc))
    return total_acc


def countfiles(dir):
    return(len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Gender-Gender reweight')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=500, metavar='N',
                        help='input batch size for testing (default: 500)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--verbose', action='store_false', default=False,
                        help='output detailed information during training, (default: False)')
    parser.add_argument('--lr', type=float, default=0.03, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--data', 
                        help='data directory')
    parser.add_argument('--nn', 
                        help='file of the reweighting nn')
    parser.add_argument('--save-to', 
                        help='save files directory')
    parser.add_argument('--vanilla', action='store_true', default=False,
                        help='All weights are 1')
    parser.add_argument('--exact', action='store_true', default=False,
                        help='Exact weights given by inverse frequencies')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    random.seed(args.seed)

    image_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

    imagedir = args.data
    train_f_dir = imagedir+"/train/female"
    train_m_dir = imagedir+"/train/male"
    test_f_dir = imagedir+"/test/female"
    test_m_dir = imagedir+"/test/male"
    #remove last / from directory
    if imagedir[-1]=="/":
        imagedir=imagedir[:-1]
    print(imagedir)


    train_data = datasets.ImageFolder(root=imagedir+"/train", transform=image_transform)

    if args.vanilla:
        version="VANILLA"
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.exact:
        freqs_train = np.array([countfiles(train_f_dir), countfiles(train_m_dir)])
        freqs_test = np.array([countfiles(test_f_dir), countfiles(test_m_dir)])
        print(freqs_train)
        print(freqs_test)
        ratios = freqs_test/freqs_train
        weights = (args.batch_size/sum(ratios)) * freqs_test/freqs_train
        print(weights)
        version = "EXACT"
        sampler = ImbalancedDatasetSampler(train_data, weights)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, **kwargs, sampler=sampler)
    else:
        weight_nn = AlexWeightNN()
        weight_nn.load_state_dict(torch.load(args.nn))
        weight_nn.to(device)
        version = "NN"
        sampler = NNSampler(train_data, weight_nn)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, **kwargs, sampler=sampler)

    out_string = version +"_"+ os.path.basename(imagedir)+"_lr="+ str(args.lr) +"_E="+str(args.epochs)
    sys.stdout = open(args.save_to+"/LOG_SAMPLER_"+out_string+".txt", 'w')


    new_freqs=[0., 0.]
    for batch_idx, (_, target) in enumerate(train_loader):
        target = target.to(device)
        new_freqs[0] += torch.sum(target == 0).item()
        new_freqs[1] += torch.sum(target == 1).item()

    print("New Frequencies {}".format(new_freqs))
    
    test_data = datasets.ImageFolder(root=imagedir+"/test", transform=image_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=500, shuffle=True, **kwargs)

    # fresh alexnet
    model = models.alexnet(num_classes=2).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    acc = 0.
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        new_acc = test(args, model, device, test_loader, 2)
        if new_acc > acc:
            acc = new_acc
            best_nn = copy.deepcopy(model)

        print('\n')

    torch.save(best_nn.state_dict(),args.save_to+"/NN_"+ out_string +".pt")
    print('Best accuracy achieved: {:.4f}\n' .format(acc))

if __name__ == '__main__':
    main()
