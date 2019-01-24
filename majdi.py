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
from classify import *
from afc import *




def main():
    start_time = time.time()
    # Pre-sets
    dtype = torch.float # not sure what this does
    #device = torch.device('cpu')
    # Globals:
    BATCH_SIZE = 25
    MAX_EPOCH = 50000
    DEVICE = torch.device('cuda')
    SEED = 7
    EARLY_STOPPING = 100
    NUM_RUNS = 10
    BALANCE_DATASET = False
    CONTRASTS_STR = []
    NETWORK = 'vgg_fine_tune'
    FINETUNE_LAYER = 12
    EXPERIMENT_NAME = 'vgg_unbalanced_4mm_fineTune_' + FINETUNE_LAYER
    LESION_SIZE = '4mm'
    SAVE_PLOTS = True
    SHOW_PLOTS = False

    if LESION_SIZE == '4mm':
        CONTRASTS_STR = ['0.91', '0.93', '0.95']
    elif LESION_SIZE == '6mm':
        CONTRASTS_STR = ['0.95', '0.97', '0.99']




    print('\nBATCH_SIZE: {}'.format(BATCH_SIZE),
          '\nMAX_EPOCH: {}'.format(MAX_EPOCH),
          '\nDEVICE: {}'.format(DEVICE),
          '\nSEED: {}'.format(SEED),
          '\nEARLY_STOPPING: {}'.format(EARLY_STOPPING),
          '\nNUM_RUNS: {}'.format(NUM_RUNS),
          '\nBALANCE_DATASET: {}'.format(BALANCE_DATASET),
          '\nLESION_SIZE: {}'.format(LESION_SIZE),
          '\nCONTRASTS_STR: {}'.format(CONTRASTS_STR),
          '\nNETWORK: {}'.format(NETWORK),
          '\nSAVE_PLOT: {}'.format(SAVE_PLOTS),
          '\nSHOW_PLOTS: {}\n'.format(SHOW_PLOTS))



    now = datetime.datetime.now()
    tmp = '/vol/research/mammo/mammo2/will/python/pyTorch/majdi/matplotlib/'
    #tmp = '/vol/vssp/cvpwrkspc01/scratch/wm0015/python_quota/matplotlib/'
    test_name = ('(' + str(NUM_RUNS) + ')_' +
    EXPERIMENT_NAME)
    #')_TL_aug_noTri_adam_0-1_fullClassifier_acc')
    #test_name = 'deleme'
    # Note - set SAVE_DIR to None to avoid saving of figures
    SAVE_DIR = tmp + '{}-{}_{}:{}_'.format(now.month, now.day, now.hour,
                                          now.minute) + test_name
    #SAVE_DIR = None

    datasets = load_data_set(0.8, DEVICE, SEED, LESION_SIZE)
    print(test_name)
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
                                      num_workers=4)


    #print_samples(dataloaders['train'], block=True, num_rows=2, num_cols=3)
    #print_samples(dataloaders['val'], block=True, num_rows=2, num_cols=3)
    #print_samples(dataloaders['test'], block=True, num_rows=2, num_cols=3)

    # Pass model single image so that it can calculate the correct shapse
    # of the layers
    sample = next(iter(dataloaders['train']))['image']
    print('sample.shape: {}'.format(sample.shape))

    roc_stats_template = {
        'train':{
            'fpr':None,
            'tpr':None},
        'val':{
            'fpr':None,
            'tpr':None},
        'test':{
            'fpr':None,
            'tpr':None}}

    stats_template = {
        'train':{
            'auc':None,
            'sens':None,
            'spec':None,
            'acc':None},
        'val':{
            'auc':None,
            'sens':None,
            'spec':None,
            'acc':None},
        'test':{
            'auc':None,
            'sens':None,
            'spec':None,
            'acc':None}}
    roc_stats = []
    stats = []

    run_num = 0
    tmp_classifications = []
    while run_num < NUM_RUNS:
        # so whats going on
        # 
        # loader dataset for current run
        datasets = load_data_set(0.8, DEVICE, SEED, LESION_SIZE,
                                 i_split=run_num,
                                 balance_dataset = BALANCE_DATASET) # latest 
        for key in dataloaders:                                      # latest
            dataloaders[key] = DataLoader(datasets[key],             # latest
                                          batch_size=BATCH_SIZE,     # latest
                                          shuffle=True,              # latest
                                          num_workers=4)             # latest
        # sample is to be an image to init the network size
        #sample = 
        if NETWORK == 'majdi':
            model = MajdiNet(sample, verbose=False)
        elif NETWORK == 'vgg':
            model = vgg19NetFullClassifier()
        elif NETWORK == 'vgg_fine_tune':
            model = vgg19NetFullClassifier_fine_tune(FINETUNE_LAYER)


        #model = vgg19NetSingleLayer()
        model = model.to(DEVICE) # Enable GPU
        # Training options
        #optimizer = optim.SGD(model.parameters(), lr=0.01)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.MSELoss()
        #criterion = nn.CrossEntropyLoss()
        #criterion = nn.NLLLoss()
        tmp_stats = dict(stats_template)
        tmp_roc = dict(roc_stats_template)

        model,\
        tmp_stats['train']['acc'],\
        tmp_stats['val']['acc'],\
        tmp_stats['test']['acc'] = train_model(
            model, criterion, optimizer, MAX_EPOCH, DEVICE,
            datasets, dataloaders, SAVE_DIR, run_num, EARLY_STOPPING,
            show_plots=SHOW_PLOTS, save_plots=SAVE_PLOTS)

        # Get ROC curve stats
        for phase in stats_template:
            tmp_roc[phase]['fpr'],\
            tmp_roc[phase]['tpr'],\
            tmp_stats[phase]['auc'],\
            tmp_stats[phase]['sens'],\
            tmp_stats[phase]['spec'] = roc_curve(
                model, DEVICE, dataloaders[phase])
        roc_stats.append(copy.deepcopy(tmp_roc))
        stats.append(copy.deepcopy(tmp_stats))
        # Only increment if network converged
        if MAX_EPOCH < 10:
            re_run_cutoff = 0
        else:
            re_run_cutoff = 0.8
        if stats[-1]['train']['auc'] > re_run_cutoff: #!!!!!!!!!!!!!!! FIX
            run_num += 1
            # classify test images with model
            tmp_classifications.append(
                classify_images(model, dataloaders, DEVICE))


        else:
            # Did not coverge, delete stats
            print('***DID NOT CONVERGE***')
            del stats[-1]
            del roc_stats[-1]

    #Append all classifications together
    classifications = {}
    num_key_collisions = 0
    for class_batch in tmp_classifications:
        added_keys = np.asarray(list(classifications.keys()))
        for index, key in enumerate(class_batch):
            if np.sum(key == added_keys) == 0:
                classifications[key] = class_batch[key]
            else:
                print('{}  ***ERROR*** key: {} already exists'.format(
                    index, key))
                num_key_collisions += 1

    # now that we have the classifications we need to get the stats on the
    #different contrasts
    # Calculate the average accuracy and stuff for each contrast
    # create array of the different contrasts
    contrasts = {CONTRASTS_STR[0]:{},
                 CONTRASTS_STR[1]:{},
                 CONTRASTS_STR[2]:{}}
    normals = {}
    num_normals = 0
    num_lesions = 0
    for f in classifications:
        parse = f.split('_') # contrast held in element 11
        if parse[11] == CONTRASTS_STR[0] or parse[11] == CONTRASTS_STR[1] or parse[11] == CONTRASTS_STR[2]:
            #contrasts[parse[11]].append(classifications[f])
            contrasts[parse[11]][f] = classifications[f]
            num_lesions += 1
        else:
            normals[f] = classifications[f]
            num_normals += 1
    #print('contrasts:\n{}'.format(contrasts))
    print('num_normals: {}'.format(num_normals))
    print('len(normals): {}'.format(len(normals)))
    print('num_lesions: {}'.format(num_lesions))
    print('num_key_collisions: {}'.format(num_key_collisions))
    print('num_0.91: {}'.format(len(contrasts[CONTRASTS_STR[0]])))
    print('num_0.93: {}'.format(len(contrasts[CONTRASTS_STR[1]])))
    print('num_0.95: {}'.format(len(contrasts[CONTRASTS_STR[2]])))
    print('len(tmp_classifications): {}'.format(len(tmp_classifications)))
    print('len(class_batch) :{}'.format(len(class_batch)))
    print('len(classifications): {}'.format(len(classifications)))
    # Calculate accuracy for each contrast
    for key in contrasts:
        tmp_lesion_class = np.asarray(
            [contrasts[key][f]['class'] for f in contrasts[key]])
        tmp_acc = sum(tmp_lesion_class)/len(tmp_lesion_class)
        print('{} accuracy: {}'.format(key, tmp_acc))

    afc(contrasts, normals, CONTRASTS_STR)




    # Loop through and save all ROC curves
    print('Saving ROC curvers to:\n{}'.format(SAVE_DIR))
    for i in range(len(roc_stats)):
        f = plt.figure()
        plt.title('ROC Curve')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        for phase in roc_stats_template:
            plt.plot(
                roc_stats[i][phase]['fpr'],
                roc_stats[i][phase]['tpr'],
                label=('{}: (area = {:.2f} sens = {:.2f} spec = {:.2f})'.format(
                    phase,
                    stats[i][phase]['auc'],
                    stats[i][phase]['sens'],
                    stats[i][phase]['spec'])))
        plt.plot([0,1],[0,1], linestyle='dashed', color='k')
        plt.grid(True)
        plt.legend()
        if SAVE_DIR != None:
            uf.save_matplotlib_figure(
                SAVE_DIR, f, 'svg', '(' + str(i) + ')ROC')
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
    print('stats:\n{}'.format(stats))
    save_results(SAVE_DIR, stats, NUM_RUNS)

    running_time = time.time() - start_time
    print('Running time:', '{:.0f}m {:.0f}s'.format(
        running_time//60, running_time%60))
    if SHOW_PLOTS:
        plt.ioff()
        plt.show()


if __name__ == '__main__':
        main()
