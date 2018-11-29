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
from networks import *




def main():
    start_time = time.time()
    # Pre-sets
    dtype = torch.float # not sure what this does
    #device = torch.device('cpu')
    # Globals:
    BATCH_SIZE = 25
    MAX_EPOCH = 5 # Really large to force early stopping
    DEVICE = torch.device('cuda:0')
    SEED = 7
    EARLY_STOPPING = 100
    NUM_RUNS = 2
    SAVE_PLOTS = True
    SHOW_PLOTS = False

    now = datetime.datetime.now()
    tmp = '/vol/research/mammo/mammo2/will/python/pyTorch/majdi/matplotlib/'
    test_name = '(' + str(NUM_RUNS) + ')_aug_noTriangles_adam'
    # Note - set SAVE_DIR to None to avoid saving of figures
    SAVE_DIR = tmp + '{}-{}_{}:{}_'.format(now.month, now.day, now.hour,
                                          now.minute) + test_name
    SAVE_DIR = None

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
    run_num = 0
    while run_num < NUM_RUNS:
        model = MajdiNet(sample, verbose=False)
        model = model.to(DEVICE) # Enable GPU
        # Training options
        #optimizer = optim.SGD(model.parameters(), lr=0.01)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        #criterion = nn.CrossEntropyLoss()
        #criterion = nn.NLLLoss()

        train_model(model, criterion, optimizer, MAX_EPOCH, DEVICE, datasets,
                    dataloaders, SAVE_DIR, run_num, EARLY_STOPPING,
                    show_plots=SHOW_PLOTS, save_plots=SAVE_PLOTS)

        # Get ROC curve stats
        for phase in stats:
            tmp = dict(stats_template)
            tmp_roc = dict(roc_template)
            tmp_roc['fpr'], tmp_roc['tpr'], tmp['auc'], tmp['sens'],\
                tmp['spec'] = roc_curve(
                    model, DEVICE, dataloaders[phase])
            roc_stats[phase].append(tmp_roc)
            stats[phase].append(tmp)
        # Only increment if network converged
        if stats['train'][-1]['auc'] > 0.8:
            run_num += 1
        else:
            # Did not coverge, delete stats
            print('***DID NOT CONVERGE***')

    # Loop through and save all ROC curves
    for i in range(len(roc_stats['train'])):
        f = plt.figure()
        plt.title('ROC Curve')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        for phase in roc_stats:
            plt.plot(
                roc_stats[phase][i]['fpr'],
                roc_stats[phase][i]['tpr'],
                label=('{}: (area = {:.2f} sens = {:.2f} spec = {:.2f})'.format(
                    phase,
                    stats[phase][i]['auc'],
                    stats[phase][i]['sens'],
                    stats[phase][i]['spec'])))
        plt.plot([0,1],[0,1], linestyle='dashed', color='k')
        plt.grid(True)
        plt.legend()
        if SAVE_DIR != None:
            uf.save_matplotlib_figure(
                SAVE_DIR, f, 'svg', '(' + str(i) + ')ROC_curve')
        if SHOW_PLOTS:
            plt.ion()
            plt.pause(0.001)



   # # Display ROC curve of first run
   # f = plt.figure()
   # plt.title('ROC Curve')
   # plt.xlabel('False positive rate')
   # plt.ylabel('True positive rate')
   # for phase in stats:
   #     plt.plot(roc_stats[phase][0]['fpr'], roc_stats[phase][0]['tpr'], label=(
   #         '{}: (area = {:.2f} sens = {:.2f} spec = {:.2f})'.format(
   #             phase, stats[phase][0]['auc'], stats[phase][0]['sens'],
   #             stats[phase][0]['spec'])))
   # plt.plot([0,1],[0,1], linestyle='dashed', color='k')
   # plt.grid(True)
   # plt.legend()

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
