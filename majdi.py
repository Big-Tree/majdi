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
import datetime
import sys
sys.path.append('/vol/research/mammo/mammo2/will/python/usefulFunctions')
import usefulFunctions as uf
from majdiFunctions import *
from dataset import *
from train import *
from rocCurve import *




# Reproducing Majdi's work with his network architecture
# init with an image to compute correct sizes
class Net(nn.Module):
    def __init__(self, image, verbose=False):
        super(Net, self).__init__()
        self.verbose = verbose
        self.save_softmax = False
        print('Net__init image.shape: {}'.format(image.shape))
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 32, 2, padding=1)
        x = self.conv1(image)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, padding=1)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, padding=1)
        self.conv3 = nn.Conv2d(64, 96, 2, padding=1)
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=2, padding=1)
        self.conv4 = nn.Conv2d(96, 128, 2, padding=1)
        x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        x = F.max_pool2d(F.relu(self.conv5(x)), kernel_size=2, padding=1)
        #self.fc1 = nn.Linear(256 * 15 * 15, 2)
        self.fc1 = nn.Linear(x.shape[1]*x.shape[2]*x.shape[3], 2)
        print('Net__init__: {}'.format(x.shape))

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
    MAX_EPOCH = 50000 # Really large to force early stopping
    DEVICE = torch.device('cuda:2')
    SEED = 7
    EARLY_STOPPING = 100
    NUM_RUNS = 10

    now = datetime.datetime.now()
    tmp = '/vol/research/mammo/mammo2/will/python/pyTorch/majdi/matplotlib/'
    test_name = '(' + str(NUM_RUNS) + ')_aug_noTriangles_adam'
    # Note - set SAVE_DIR to None to avoid saving of figures
    SAVE_DIR = tmp + '{}-{}_{}:{}_'.format(now.month, now.day, now.hour,
                                          now.minute) + test_name
    #SAVE_DIR = None

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

    # Build dataloaders
    dataloaders = {'train':None,
                  'val':None,
                  'test':None}
    for key in dataloaders:
        dataloaders[key] = DataLoader(datasets[key],
                                      batch_size=BATCH_SIZE,
                                      shuffle=True,
                                      num_workers=1)

    #print_samples(dataloaders['train'], block=True, num_rows=2, num_cols=3)
    #print_samples(dataloaders['val'], block=True, num_rows=2, num_cols=3)
    #print_samples(dataloaders['test'], block=True, num_rows=2, num_cols=3)

    # Pass model single image so that it can calculate the correct shapse
    # of the layers
    sample = next(iter(dataloaders['train']))['image']
    print('sample.shape: {}'.format(sample.shape))
    stats_template = {
        'auc':[],
        'sens':[],
        'spec':[]}
    roc_template = {
        'fpr': None,
        'tpr': None}
    roc_stats = {
        'train':[],
        'val':[],
        'test':[]}
    stats = {
        'train':[],
        'val':[],
        'test':[]}
    for run_num in range(NUM_RUNS):
        model = Net(sample, verbose=False)
        model = model.to(DEVICE) # Enable GPU
        # Training options
        #optimizer = optim.SGD(model.parameters(), lr=0.01)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        #criterion = nn.CrossEntropyLoss()
        #criterion = nn.NLLLoss()

        train_model(model, criterion, optimizer, MAX_EPOCH, DEVICE, datasets,
                    dataloaders, SAVE_DIR, run_num, EARLY_STOPPING,
                    show_plots=False, save_plots=True)

        # Get ROC curve stats
        for phase in stats:
            tmp = dict(stats_template)
            tmp_roc = dict(roc_template)
            tmp_roc['fpr'], tmp_roc['tpr'], tmp['auc'], tmp['sens'],\
                tmp['spec'] = roc_curve(
                    model, DEVICE, dataloaders[phase])
            roc_stats[phase].append(tmp_roc)
            stats[phase].append(tmp)

    # Display ROC curve of first run
    f = plt.figure()
    plt.title('ROC Curve')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    for phase in stats:
        plt.plot(roc_stats[phase][0]['fpr'], roc_stats[phase][0]['tpr'], label=(
            '{}: (area = {:.2f} sens = {:.2f} spec = {:.2f})'.format(
                phase, stats[phase][0]['auc'], stats[phase][0]['sens'],
                stats[phase][0]['spec'])))
    plt.plot([0,1],[0,1], linestyle='dashed', color='k')
    plt.grid(True)
    plt.legend()

    # Display and save stats for runs
    save_results(SAVE_DIR, stats, NUM_RUNS)

    running_time = time.time() - start_time
    print('Running time:', '{:.0f}m {:.0f}s'.format(
        running_time//60, running_time%60))
    if SAVE_DIR != None:
        # Save figure
        uf.save_matplotlib_figure(SAVE_DIR, f, 'svg', 'ROC')
    plt.ioff()
    plt.show()


if __name__ == '__main__':
        main()
