import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # gradient descent
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


# Class to read the dicom images in for training and testing of the network
class LoadDataSet():
    def __init__(self, split_ratio, batch_size, device):
        # Initialise class variables:
        self.batch_number = 0 # Keeps track of the minibatches during training
        self.batch_size = batch_size
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
        self.train = {
            'data': np.asarray([_['image'] for _ in dataset_mixer][0 : s_p]),
            'labels': np.asarray([_['class'] for _ in dataset_mixer][0 : s_p])}
        self.val = {
            'data': np.asarray(
                [_['image'] for _ in dataset_mixer][s_p : s_p_val_test]),
            'labels': np.asarray(
                [_['class'] for _ in dataset_mixer][s_p : s_p_val_test])}
        self.test = {
            'data': np.asarray(
                [_['image'] for _ in dataset_mixer][s_p_val_test : ]),
            'labels': np.asarray(
                [_['class'] for _ in dataset_mixer][s_p_val_test : ])}
        # Convert to one hot labels
        tmp_labels = self.train['labels']
        self.train['labels'] = np.zeros((len(tmp_labels), 2))
        self.train['labels'][range(len(tmp_labels)), tmp_labels] = 1

        tmp_labels = self.val['labels']
        self.val['labels'] = np.zeros((len(tmp_labels), 2))
        self.val['labels'][range(len(tmp_labels)), tmp_labels] = 1

        tmp_labels = self.test['labels']
        self.test['labels'] = np.zeros((len(tmp_labels), 2))
        self.test['labels'][range(len(tmp_labels)), tmp_labels] = 1

        # Reshape the images
        tmp = self.train['data'].shape
        self.train['data'].shape = (tmp[0], 1, tmp[1], tmp[2])
        tmp = self.val['data'].shape
        self.val['data'].shape = (tmp[0], 1, tmp[1], tmp[2])
        tmp = self.test['data'].shape
        self.test['data'].shape = (tmp[0], 1, tmp[1], tmp[2])

        # Normalise between -1 and 1
        # Get max value
        max_pixel = np.amax([np.amax(self.train['data']),
                             np.amax(self.val['data']),
                             np.amax(self.test['data'])])
        print('Max pixel: ', max_pixel)
        self.train['data'] = self.train['data']/max_pixel*2 - 1
        self.val['data'] = self.val['data']/max_pixel*2 - 1
        self.test['data'] = self.test['data']/max_pixel*2 - 1
        print('min self.train: ', np.amin(self.train['data']))
        print('max self.train: ', np.amax(self.train['data']))
        print('min self.val: ', np.amin(self.val['data']))
        print('max self.val: ', np.amax(self.val['data']))
        print('min self.test: ', np.amin(self.test['data']))
        print('max self.test: ', np.amax(self.test['data']))


    def show_images(self):
      pass


    def get_batch_train(self, verbose = False):
        self.batch_number += 1
        indices = range(
            (self.batch_number-1)*self.batch_size,
            self.batch_number*self.batch_size)
        if verbose == True: print('indices: ', indices)
        out = self.train['data'].take(indices, mode='wrap', axis=0)
        #out = out.astype(np.float64)
        if verbose == True: print('out.shape: ', out.shape)
        if verbose == True: print('out.dtype: ', out.dtype)
        out = torch.from_numpy(out).float()
        out = out.to(DEVICE)
        if verbose == True: print('out.shape_post: ', out.shape)
        return out
    def get_batch_train_all(self):
        out = self.train['data']
        out = torch.from_numpy(out).float()
        out = out.to(DEVICE)
        return out
    def get_labels_train(self, verbose = False):
        indices = range(
            (self.batch_number-1)*self.batch_size,
            self.batch_number*self.batch_size)
        out = self.train['labels'].take(indices, mode='wrap', axis=0)
        out = torch.from_numpy(out).float()
        out = out.to(DEVICE)
        if verbose == True: print('out.dtype: ', out.dtype)
        return out
    def get_labels_train_all(self):
        out = self.train['labels']
        out = torch.from_numpy(out).float()
        out = out.to(DEVICE)
        return out
    def get_batch_val(self):
        indices = range(
            (self.batch_number-1)*self.batch_size,
            self.batch_number*self.batch_size)
        out = self.val['data'].take(indices, mode='wrap', axis=0)
        #out = out.astype(np.float64)
        out = torch.from_numpy(out).float()
        out = out.to(DEVICE)
        return out
    def get_batch_val_all(self):
        out = self.val['data']
        out = torch.from_numpy(out).float()
        out = out.to(DEVICE)
        return out
    def get_labels_val(self):
        indices = range(
            (self.batch_number-1)*self.batch_size,
            self.batch_number*self.batch_size)
        out = self.val['labels'].take(indices, mode='wrap', axis=0)
        out = torch.from_numpy(out).float()
        out = out.to(DEVICE)
        return out
    def get_labels_val_all(self):
        out = self.val['labels']
        out = torch.from_numpy(out).float()
        out = out.to(DEVICE)
        return out
    def get_batch_test_all(self):
        out = self.test['data']
        out = torch.from_numpy(out).float()
        out = out.to(DEVICE)
        return out
    def get_labels_test_all(self):
        out = self.test['labels']
        out = torch.from_numpy(out).float()
        out = out.to(DEVICE)
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
        #x = F.softmax(x, dim=1)
        m = nn.LogSoftmax(dim=1)
        x = m(x)
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
STEPS = 6000
DEVICE = torch.device('cuda:2')
STATS_STEPS = int(STEPS/100)
SEED = 7
if STATS_STEPS <= 1:
    STATS_STEPS = 5 # Every 5 steps get loss and accuracy stats

dataset = LoadDataSet(0.8, BATCH_SIZE, DEVICE)
print('dataset.train[data].shape: ', dataset.train['data'].shape)
print('dataset.val[data].shape: ', dataset.val['data'].shape)
print('dataset.test[data].shape: ', dataset.test['data'].shape)
print('dataset.train[labels].shape: ', dataset.train['labels'].shape)
print('dataset.val[labels].shape: ', dataset.val['labels'].shape)
print('dataset.test[labels].shape: ', dataset.test['labels'].shape)
model = Net(verbose=False)
model = model.to(DEVICE) # Enable GPU
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
for step_counter in range(STEPS):
    print('Step (', step_counter, '/', STEPS, ')')
    if step_counter % STATS_STEPS == 0:
        # Calculate training accuracy and loss on all train images
        print('    TRAIN STATS...')
        tmp_acc, tmp_loss = (
            get_stats_epoch(
                model,
                criterion,
                dataset.get_batch_train_all(),
                dataset.get_labels_train_all(),
                25))
        accuracies['smooth']['train'].append(tmp_acc)
        losses['smooth']['train'].append(tmp_loss)

        # Calculate validation accuracy and loss based on all val images
        print('    VAL STATS...')
        tmp_acc, tmp_loss = (
            get_stats_epoch(
                model,
                criterion,
                dataset.get_batch_val_all(),
                dataset.get_labels_val_all(),
                25))
        accuracies['smooth']['val'].append(tmp_acc)
        losses['smooth']['val'].append(tmp_loss)
        # Save if best model
        if tmp_loss < model_best['loss']:
            model_best['loss'] = tmp_loss
            model_best['step'] = step_counter
            model_best['model'].load_state_dict(model.state_dict())
            print('****New best model found :D')

        # Calculate test accuracy and loss based on all val images
        print('    TEST STATS...')
        tmp_acc, tmp_loss = (
            get_stats_epoch(
                model,
                criterion,
                dataset.get_batch_test_all(),
                dataset.get_labels_test_all(),
                25))
        accuracies['smooth']['test'].append(tmp_acc)
        losses['smooth']['test'].append(tmp_loss)

    print('    OPTIMISE...')
    # Calculate loss and accuracy for single step
    # get data for step
    batch = dataset.get_batch_train()
    batch_val = dataset.get_batch_val()
    optimizer.zero_grad()
    model.save_softmax = True # <-----------------------------------------
    output = model(batch)
    labels_train = dataset.get_labels_train()
    labels_val = dataset.get_labels_val()
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

plt.show()
