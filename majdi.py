import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # gradient descent
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
import random
import pydicom
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


# Class to read the dicom images in for training and testing of the network
# Function to read in the DICOM images, convert to numpy arrays, label and then
# loads the images into three classes for train, val and test.  These classes,
# which are returned will be used in the network main loop.
# A split ratio of 0.8 will set 80% of the images for training, 10% for
# validaiton and 10% for test
def load_data_set(split_ratio, batch_size, device):
    # Initialise class variables:
    # Load in the dicom files
    # Get file list
    print('Getting file list...')
    file_list = {
        'backgrounds': uf.get_files(
            '/vol/research/mammo/mammo2/will/data/prem/Segments',
            '2D_dim2d.dcm'),
        'lesions': uf.get_files(
            '/vol/research/mammo/mammo2/will/data/prem/2D/6mm',
            '*.dcm')}
    # Balance the dataset
    file_list['backgrounds'] = file_list['backgrounds'][
        0 : len(file_list['lesions'])]
    # Load in dicom images to RAM
    dicom_images = {'backgrounds':[], 'lesions':[]}
    for key in file_list:
        print('Loading dicom', key, '...')
        for index, f in enumerate(file_list[key]):
            dicom_images[key].append(pydicom.dcmread(f))
            print('    ', index, '/', len(file_list[key]))
    rgb_images = {'backgrounds':[], 'lesions':[]}
    print('Converting dicom to RGB...')
    for key in rgb_images:
        for img in dicom_images[key]:
            rgb_images[key].append(img.pixel_array)

    # Split the data into training and val
    # Mix the lesions and backgrounds
    # [[img, label], [img, label]]
    # Note - dictionaries preserve insertion order
    dataset_mixer = []
    it = iter(rgb_images)
    print('class 0: {}'.format(next(it)))
    print('class 1: {}'.format(next(it)))
    for index, key in enumerate(rgb_images):
        for img in rgb_images[key]:
            dataset_mixer.append({
                'image': img,
                'class': index})
    if SEED != None:
        random.seed(SEED) # Fix datasets
    random.shuffle(dataset_mixer)
    s_p = round(split_ratio*len(dataset_mixer))
    e_p = len(dataset_mixer)
    s_p_val_test = round((s_p+e_p)/2)
    train = {
        'data': np.asarray([_['image'] for _ in dataset_mixer][0 : s_p]),
        'labels': np.asarray([_['class'] for _ in dataset_mixer][0 : s_p])}
    val = {
        'data': np.asarray(
            [_['image'] for _ in dataset_mixer][s_p : s_p_val_test]),
        'labels': np.asarray(
            [_['class'] for _ in dataset_mixer][s_p : s_p_val_test])}
    test = {
        'data': np.asarray(
            [_['image'] for _ in dataset_mixer][s_p_val_test : ]),
        'labels': np.asarray(
            [_['class'] for _ in dataset_mixer][s_p_val_test : ])}
    # Convert to one hot labels
    tmp_labels = train['labels']
    train['labels'] = np.zeros((len(tmp_labels), 2))
    train['labels'][range(len(tmp_labels)), tmp_labels] = 1

    tmp_labels = val['labels']
    val['labels'] = np.zeros((len(tmp_labels), 2))
    val['labels'][range(len(tmp_labels)), tmp_labels] = 1

    tmp_labels = test['labels']
    test['labels'] = np.zeros((len(tmp_labels), 2))
    test['labels'][range(len(tmp_labels)), tmp_labels] = 1

    # Reshape the images
    tmp = train['data'].shape
    train['data'].shape = (tmp[0], 1, tmp[1], tmp[2])
    tmp = val['data'].shape
    val['data'].shape = (tmp[0], 1, tmp[1], tmp[2])
    tmp = test['data'].shape
    test['data'].shape = (tmp[0], 1, tmp[1], tmp[2])

    # Normalise between -1 and 1
    # Get max value
    max_pixel = np.amax([np.amax(train['data']),
                         np.amax(val['data']),
                         np.amax(test['data'])])
    print('Max pixel: ', max_pixel)
    train['data'] = train['data']/max_pixel*2 - 1
    val['data'] = val['data']/max_pixel*2 - 1
    test['data'] = test['data']/max_pixel*2 - 1
    print('min train: ', np.amin(train['data']))
    print('max train: ', np.amax(train['data']))
    print('min val: ', np.amin(val['data']))
    print('max val: ', np.amax(val['data']))
    print('min test: ', np.amin(test['data']))
    print('max test: ', np.amax(test['data']))

    # Load the images into the dataset class
    out = {
        'train':MajdiDataset(train['data'],
                             train['labels'],
                             transform=ToTensor()),
        'val':MajdiDataset(val['data'],
                           val['labels'],
                           transform=ToTensor()),
        'test':MajdiDataset(test['data'],
                            test['labels'],
                            transform=ToTensor())}

    return out


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


start_time = time.time()
# Pre-sets
dtype = torch.float # not sure what this does
#device = torch.device('cpu')
# Globals:
BATCH_SIZE = 25
MAX_EPOCH = 100
STEPS = 6000
DEVICE = torch.device('cuda:2')
STATS_STEPS = int(STEPS/100)
SEED = 7
if STATS_STEPS <= 1:
    STATS_STEPS = 5 # Every 5 steps get loss and accuracy stats

datasets = load_data_set(0.8, BATCH_SIZE, DEVICE)
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
print('model.parameters().is_cuda(): {}'.format(next(model.parameters()).is_cuda))
print('model.parameters().device: {}'.format(next(model.parameters()).device))
input = torch.randn(1, 1, 210, 210, device = DEVICE)
input = torch.randn(1, 1, 211, 211, device = DEVICE)
input = torch.randn(1, 1, 429, 429, device = DEVICE)
output = model(input)


optimizer = optim.SGD(model.parameters(), lr=0.01)
#optimizer = optim.Adam(model.parameters())

criterion = nn.MSELoss()
#criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()


losses = {'step':{'train':[],
                  'val':[]},
          'smooth':{'train':[],
                    'val':[],
                    'test':[]}}
accuracies = {'step':{'train':[], # Check that this variable *********
                      'val':[]},
              'smooth':{'train':[],
                        'val':[],
                        'test':[]}}
model_best = {'loss':999,
              'step':0,
              'model':Net().to(DEVICE)}
dataloader = {'train': DataLoader(datasets['train'],
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=4),
              'val': DataLoader(datasets['val'],
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=4),
              'test': DataLoader(datasets['test'],
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=4)}
modes = ['train', 'val', 'test']

# Print some of the images
#inputs, classes = next(iter(dataloader['train']))
data_dict = next(iter(dataloader['train']))
inputs = data_dict['image']

train_model(model, criterion, optimizer, MAX_EPOCH, DEVICE, datasets,
            dataloader)

exit()




if 1 == 2:
    for i_epoch in range(MAX_EPOCH):
        for phase in modes:
            # Training epoch
            #for inputs, labels in dataloader[phase]:
            for data_dict in dataloader[phase]:
                inputs = data_dict['image']
                labels = data_dict['label']
                print('inputs: {}'.format(inputs))
                print('labels: {}'.format(labels))
                inputs = inputs.to(DEVICE, dtype=torch.float)
                labels = labels.to(DEVICE, dtype=torch.float)
                print('inputs: {}'.format(inputs))
                print('labels: {}'.format(labels))
                optimizer.zero_grad()
                print('    model.parameters().device: {}'.format(next(model.parameters()).device))
                print()
                output = model(inputs)
                # Calculate accuracy
                output = round(output)
                output = round(sample_batch['label'].data.cpu().numpy()) # (n,2)
                labels = sample_batch['label'].data.cpu().numpy()
                matches = [output[:,0] == labels[:, 0]]
                accuracy = sum(matches)/len(matches)
                accuracies['step']['train'].append(accuracy)

                # Loss
                loss = criterion(output, sample_batch['label'])
                losses['step']['train'].append(loss.item())

                # Optimise
                loss.backwards()
                optimiser.step()






if 1 == 2:
    for step_counter in range(STEPS):
        print('Step (', step_counter, '/', STEPS, ')')
        #if step_counter % STATS_STEPS == 0:
           # # Calculate training accuracy and loss on all train images
           # print('    TRAIN STATS...')
           # tmp_acc, tmp_loss = (
           #     get_stats_epoch(
           #         model,
           #         criterion,
           #         [_['data'] for _ in train], # all train data
           #         [_['labels'] for _ in train], # all train labels
           #         25))
           # accuracies['smooth']['train'].append(tmp_acc)
           # losses['smooth']['train'].append(tmp_loss)

           # # Calculate validation accuracy and loss based on all val images
           # print('    VAL STATS...')
           # tmp_acc, tmp_loss = (
           #     get_stats_epoch(
           #         model,
           #         criterion,
           #         [_['data'] for _ in val], # all val data
           #         [_['labels'] for _ in val], # all val labels
           #         25))
           # accuracies['smooth']['val'].append(tmp_acc)
           # losses['smooth']['val'].append(tmp_loss)
           # # Save if best model
           # if tmp_loss < model_best['loss']:
           #     model_best['loss'] = tmp_loss
           #     model_best['step'] = step_counter
           #     model_best['model'].load_state_dict(model.state_dict())
           #     print('****New best model found :D')

           # # Calculate test accuracy and loss based on all val images
           # print('    TEST STATS...')
           # tmp_acc, tmp_loss = (
           #     get_stats_epoch(
           #         model,
           #         criterion,
           #         [_['data'] for _ in test], # all val data
           #         [_['labels'] for _ in test], # all val labels
           #         25))
           # accuracies['smooth']['test'].append(tmp_acc)
           # losses['smooth']['test'].append(tmp_loss)

        print('    OPTIMISE...')

        # Calculate loss and accuracy for single step
        # get data for step
        tmp = train_dataloader.next()
        batch_train = tmp['data']
        labels_train = tmp['lables']
        tmp = val_dataloader.next()
        batch_val = tmp['data']
        labels_val = tmp['labels']

        optimizer.zero_grad()
        output = model(batch_train)
        # Calculate accuracy and track
        pred = np.zeros((len(labels_train), 2))
        maxOutput = [np.argmax(_) for _ in output.data.cpu().numpy()]
        pred[range(BATCH_SIZE), maxOutput] = 1
        accuracy = sum(pred == labels_train.cpu().numpy())/len(labels_train)
        accuracy = accuracy[0]
        accuracies['step']['train'].append(accuracy)

        # Track losses
        loss = criterion(output, labels_train)
        losses['step']['train'].append(loss.item())

        # Calculate the validation accuracy and track
        output = model(batch_val)
        pred = np.zeros((len(labels_val), 2))
        maxOutput = [np.argmax(_) for _ in output.data.cpu().numpy()]
        pred[range(BATCH_SIZE), maxOutput] = 1
        accuracy = sum(pred == labels_val.cpu().numpy())/len(labels_val)
        accuracy = accuracy[0]
        print('step val accuracy: {}'.format(accuracy))

        accuracies['step']['val'].append(accuracy)
        # Track losses
        loss_val = criterion(output, labels_val)
        losses['step']['val'].append(loss_val.item())

        # Compute gradients and optimise
        loss.backward()
        optimizer.step()

# Plot loss
plt.figure()
plt.title('Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
#plt.plot(range(len(losses['step']['train'])), losses['step']['train'], label='train_step')
#plt.plot(range(len(losses['step']['val'])), losses['step']['val'], label='val_step')
plt.plot(range(0, len(losses['smooth']['train'])*STATS_STEPS, STATS_STEPS),
         losses['smooth']['train'],
         label='Train')
plt.plot(range(0, len(losses['smooth']['val'])*STATS_STEPS, STATS_STEPS),
         losses['smooth']['val'],
         label='Validation')
plt.plot(range(0, len(losses['smooth']['test'])*STATS_STEPS, STATS_STEPS),
         losses['smooth']['test'],
         label='Test')
plt.axvline(x=model_best['step'],  linestyle='dashed', color='k')
plt.grid(True)
plt.legend()

# Plot accuracy
plt.figure()
plt.title('Accuracy')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.plot(range(0, len(accuracies['smooth']['train'])*STATS_STEPS, STATS_STEPS),
         accuracies['smooth']['train'],
         label='Train')
plt.plot(range(0, len(accuracies['smooth']['val'])*STATS_STEPS, STATS_STEPS),
         accuracies['smooth']['val'],
         label='Validation')
plt.plot(range(0, len(accuracies['smooth']['test'])*STATS_STEPS, STATS_STEPS),
         accuracies['smooth']['test'],
         label='Test')
plt.axvline(x=model_best['step'],  linestyle='dashed', color='k')
#plt.plot(range(len(accuracies['step']['train'])), accuracies['step']['train'], label='train_step')
#plt.plot(range(len(accuracies['step']['val'])), accuracies['step']['val'], label='val_step')
plt.grid(True)
plt.legend()

# ROC Training
fpr_train, tpr_train, auc_train= get_roc_curve(
    model,
    dataset.get_batch_train_all(),
    dataset.get_labels_train_all(),
    optimizer,
    10)
# ROC Validation
fpr_val, tpr_val, auc_val= get_roc_curve(
    model,
    dataset.get_batch_val_all(),
    dataset.get_labels_val_all(),
    optimizer,
    10)
# ROC Test
fpr_test, tpr_test, auc_test= get_roc_curve(
    model,
    dataset.get_batch_test_all(),
    dataset.get_labels_test_all(),
    optimizer,
    10)
# ROC model_best
fpr_best, tpr_best, auc_best= get_roc_curve(
    model_best['model'],
    dataset.get_batch_test_all(),
    dataset.get_labels_test_all(),
    optimizer,
    10)
plt.figure()
plt.title('ROC Curve - Training, Validation & Test')
plt.plot(fpr_train, tpr_train, label='Train (area = {:.2f})'.format(auc_train))
plt.plot(fpr_val, tpr_val, label='Validation (area = {:.2f})'.format(auc_val))
plt.plot(fpr_test, tpr_test, label='Test (area = {:.2f})'.format(auc_test))
plt.plot(fpr_best, tpr_best, label='Model_best (area = {:.2f})'.format(auc_best))
plt.plot([0,1],[0,1], linestyle='dashed', color='k')
plt.grid(True)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend()

params = list(model.parameters())
print(len(params))
for _ in params:
    print(_.size())
#print(params[0].size())
#print(model)

print('Running time:', '{:.2f}'.format(time.time() - start_time), ' s')
plt.ioff() # Turn interactive off so that plt.show blocks
plt.show()
