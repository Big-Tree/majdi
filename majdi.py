import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # gradient descent
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import sys
sys.path.append('/vol/research/mammo/mammo2/will/python/usefulFunctions')
import usefulFunctions as uf
from majdiFunctions import *
from dataset import *
from train import *
from rocCurve import *




# Reproducing Majdi's work with his network architecture
class Net(nn.Module):
    def __init__(self, verbose=False):
        super(Net, self).__init__()
        self.verbose = verbose
        self.save_softmax = False
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 32, 2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, padding=1)
        self.conv3 = nn.Conv2d(64, 96, 2, padding=1)
        self.conv4 = nn.Conv2d(96, 128, 2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(256 * 15 * 15, 2)

    def forward(self, x):
        if self.verbose == True: print('forward, x.type: ', x.type())
        if self.verbose == True: print('0: ', x.shape)
        x = self.conv1(x)
        if self.verbose == True: print('1: ', x.shape)
        x = F.relu(x)
        if self.verbose == True: print('2: ', x.shape)
        x = F.max_pool2d(x, kernel_size=2, padding=1)
        if self.verbose == True: print('3: ', x.shape)
        #x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, padding=1)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, padding=1)
        if self.verbose == True: print('4: ', x.shape)
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=2, padding=1)
        x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=2, padding=1)
        if self.verbose == True: print('5: ', x.shape)
        x = F.max_pool2d(F.relu(self.conv5(x)), kernel_size=2, padding=1)

        if self.verbose == True: print('6: ', x.shape)
        x = x.view(-1, self.num_flat_features(x))
        if self.verbose == True: print('7: ', x.shape)
        x = self.fc1(x)
        if self.verbose == True: print('8: ', x.shape)
        x = F.softmax(x, dim=1)
        #m = nn.LogSoftmax(dim=1)
        #x = m(x)
        if self.verbose == True: print('9: ', x.shape)
        if self.save_softmax == True:
            self.save_softmax = False
            self.softmax_out = x
        return x

    # Gets the number of features in the layer (calculates the shape of a
    # flatten operation)
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def main():
    start_time = time.time()
    # Pre-sets
    dtype = torch.float # not sure what this does
    #device = torch.device('cpu')
    # Globals:
    BATCH_SIZE = 25
    MAX_EPOCH = 200
    DEVICE = torch.device('cuda:2')
    SEED = 7

    plt.ion()
    datasets = load_data_set(0.8, DEVICE, SEED)
    print('len(datasets[train]): {}'.format(len(datasets['train'])))
    print('len(datasets[val]): {}'.format(len(datasets['val'])))
    print('len(datasets[test]): {}'.format(len(datasets['test'])))
    tmp = datasets['train'][0]['image'].shape
    print('tmp: {}'.format(tmp))
    #tmp = [_['image'] for _ in datasets['train']
    print('datasets[train][0][image].shape: {}'.format(
        datasets['train'][0]['image'].shape))
    print('datasets[val][0][image].shape: {}'.format(
        datasets['val'][0]['image'].shape))
    print('datasets[test][0][image].shape: {}'.format(
        datasets['test'][0]['image'].shape))
    print('datasets[train][0][label]: {}'.format(
        datasets['train'][0]['label'].shape))
    print('datasets[val][0][label]: {}'.format(
        datasets['val'][0]['label'].shape))
    print('datasets[test][0][label]: {}'.format(
        datasets['test'][0]['label'].shape))

    # Print some of the images
    if 1 == 2:
        plt.ion()
        fig = plt.figure()
        sample = datasets['train'][0]
        print('\nsample[image].shape: {}\nsample[label].shape: {}'.format(
            sample['image'].shape, sample['label'].shape))
        img_stacked = np.stack((sample['image'],)*3, axis=-1)
        img_stacked = np.squeeze(img_stacked)
        img_stacked = (img_stacked + 1)/2
        print('img_stacked.shape: {}'.format(img_stacked.shape))
        plt.imshow(img_stacked)
        plt.pause(0.001) # Displays the figures I think

    model = Net(verbose=False)
    model = model.to(DEVICE) # Enable GPU

    # Training options
    #optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()

    # Build dataloaders
    dataloaders = {'train':None,
                  'val':None,
                  'test':None}
    for key in dataloaders:
        dataloaders[key] = DataLoader(datasets[key],
                                      batch_size=BATCH_SIZE,
                                      shuffle=True,
                                      num_workers=4)
    modes = ['train', 'val', 'test']

    print_samples(dataloaders['train'], block=True, num_rows=4, num_cols=5)
    # Print some of the images
    #inputs, classes = next(iter(dataloaders['train']))
    data_dict = next(iter(dataloaders['train']))
    inputs = data_dict['image']

    train_model(model, criterion, optimizer, MAX_EPOCH, DEVICE, datasets,
                dataloaders)

    # ROC Curve
    phases = ['train', 'val', 'test']
    fpr = {'train': None, 'val': None, 'test': None}
    tpr = {'train': None, 'val': None, 'test': None}
    auc = {'train': None, 'val': None, 'test': None}
    sens = {'train': None, 'val': None, 'test': None}
    spec = {'train': None, 'val': None, 'test': None}

    plt.figure()
    plt.title('ROC Curve')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    for phase in phases:
        fpr[phase], tpr[phase], auc[phase], sens[phase], spec[phase] = (
            roc_curve(model, DEVICE, dataloaders[phase]))
        plt.plot(fpr[phase], tpr[phase], label=(
            '{}: (area = {:.2f} sens = {:.2f} spec = {:.2f})'.format(
                phase, auc[phase], sens[phase], spec[phase])))
    plt.plot([0,1],[0,1], linestyle='dashed', color='k')
    plt.grid(True)
    plt.legend()
    print('Running time:', '{:.2f}'.format(time.time() - start_time), ' s')
    plt.ioff()
    plt.show()


if __name__ == '__main__':
        main()
