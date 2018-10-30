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
        random.shuffle(dataset_mixer)
        s_p = round(split_ratio*len(dataset_mixer))
        self.train = {
            'data': np.asarray([_['image'] for _ in dataset_mixer][0 : s_p]),
            'labels': np.asarray([_['class'] for _ in dataset_mixer][0 : s_p])}
        self.val = {
            'data': np.asarray([_['image'] for _ in dataset_mixer][s_p : ]),
            'labels': np.asarray([_['class'] for _ in dataset_mixer][s_p : ])}
        # Convert to one hot labels
        tmp_labels = self.train['labels']
        self.train['labels'] = np.zeros((len(tmp_labels), 2))
        self.train['labels'][range(len(tmp_labels)), tmp_labels] = 1

        tmp_labels = self.val['labels']
        self.val['labels'] = np.zeros((len(tmp_labels), 2))
        self.val['labels'][range(len(tmp_labels)), tmp_labels] = 1

        # Reshape the images
        tmp = self.train['data'].shape
        self.train['data'].shape = (tmp[0], 1, tmp[1], tmp[2])
        tmp = self.val['data'].shape
        self.val['data'].shape = (tmp[0], 1, tmp[1], tmp[2])

        # Normalise between -1 and 1
        # Get max value
        max_pixel = np.amax([np.amax(self.train['data']),
                             np.amax(self.val['data'])])
        print('Max pixel: ', max_pixel)
        self.train['data'] = self.train['data']/max_pixel*2 - 1
        self.val['data'] = self.val['data']/max_pixel*2 - 1
        print('min self.train: ', np.amin(self.train['data']))
        print('max self.train: ', np.amax(self.train['data']))
        print('min self.val: ', np.amin(self.val['data']))
        print('max self.val: ', np.amax(self.val['data']))


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
        x = F.softmax(x, dim=1) # Should it be put through a relu? Is it dim 0?
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
STEPS = 1000
DEVICE = torch.device('cuda:2')
STATS_STEPS = int(STEPS/100)
if STATS_STEPS <= 1:
    STATS_STEPS = 5 # Every 5 steps get loss and accuracy stats

dataset = LoadDataSet(0.9, BATCH_SIZE, DEVICE)
print('dataset.train[data].shape: ', dataset.train['data'].shape)
print('dataset.val[data].shape: ', dataset.val['data'].shape)
print('dataset.train[labels].shape: ', dataset.train['labels'].shape)
print('dataset.val[labels].shape: ', dataset.val['labels'].shape)
net = Net(verbose=False)
net = net.to(DEVICE) # Enable GPU
input = torch.randn(1, 1, 210, 210, device = DEVICE)
input = torch.randn(1, 1, 211, 211, device = DEVICE)
input = torch.randn(1, 1, 429, 429, device = DEVICE)
output = net(input)


# Use the gradients to update the weights
#optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer = optim.Adam(net.parameters())

# Print the parameters before and after
# Before:
#print('Network params before')
#print(list(net.parameters()))
criterion = nn.MSELoss()

train_losses_step = []
val_losses_step = []
val_losses_smooth = []
train_losses_smooth = []

train_accuracies_step = []
val_accuracies_step = []
val_accuracies_smooth = []
val_accuracies_simple = [] # DELETE LINE
val_accuracies_new = [] # DELETE LINE
train_accuracies_smooth = []
train_accuracies_simple = [] # DELETE LINE
delme_accuracy = []
for i in range(STEPS):
    print('Step (', i, '/', STEPS, ')')
    if i % STATS_STEPS == 0:
        # Calculate training accuracy and loss on all train images
        print('    TRAIN STATS...')
        tmp_acc, tmp_loss, tmp_simple, tmp_new= (
            get_stats_epoch(
                net,
                criterion,
                dataset.get_batch_train_all(),
                dataset.get_labels_train_all(),
                25))
        train_accuracies_smooth.append(tmp_acc)
        train_losses_smooth.append(tmp_loss)
        train_accuracies_simple.append(tmp_simple)


        # Calculate validation accuracy and loss based on all val images
        print('    VAL STATS...')
        tmp_acc, tmp_loss, tmp_simple, tmp_new = (
            get_stats_epoch(
                net,
                criterion,
                dataset.get_batch_val_all(),
                dataset.get_labels_val_all(),
                25))
        val_accuracies_smooth.append(tmp_acc)
        val_losses_smooth.append(tmp_loss)
        val_accuracies_simple.append(tmp_simple)
        val_accuracies_new.append(tmp_new)

    print('    OPTIMISE...')
    # Calculate loss and accuracy for single step
    # get data for step
    batch = dataset.get_batch_train()
    batch_val = dataset.get_batch_val()
    optimizer.zero_grad()
    net.save_softmax = True # <-----------------------------------------
    output = net(batch)
    labels_train = dataset.get_labels_train()
    labels_val = dataset.get_labels_val()
    # Calculate accuracy and track
    pred = np.zeros((len(labels_train), 2))
    maxOutput = [np.argmax(_) for _ in output.data.cpu().numpy()]
    pred[range(BATCH_SIZE), maxOutput] = 1
    accuracy = sum(pred == labels_train.cpu().numpy())/len(labels_train)
    accuracy = accuracy[0]
    train_accuracies_step.append(accuracy)

    # Test the softmax activations by comparing the acuracies
    softmax_activations = net.softmax_out.detach().cpu().numpy()
    s_accuracy = softmax_activations.round() == labels_train.detach().cpu().numpy()
    s_accuracy = sum(s_accuracy/len(s_accuracy))
    print('Given accuracy: {}'.format(accuracy))
    print('s_accuracy: {}'.format(s_accuracy))
    # Track losses
    loss = criterion(output, labels_train)
    train_losses_step.append(loss.item())


    # val accuracy redo
    #output = net(batch_val)
    #print('output.shape: {}'.format(output.shape))
    #output = output.data.cpu().numpy() # (b, 2)
    #output = output.round()
    #print('output: {}'.format(output))
    #print('labels: {}'.format(labels_val.cpu().numpy()))
    #print('output == labels: {}'.format(output == labels_val.cpu().numpy()))
    #print('output.shape: {}'.format(output.shape))
    #accuracy = sum(output == labels_val.cpu().numpy())/len(output)
    #accuracy = accuracy[0]
    #delme_accuracy.append(accuracy)
    #print('new accuracy: {}'.format(accuracy))

    # Calculate the validation accuracy and track
    output = net(batch_val)
    pred = np.zeros((len(labels_val), 2))
    maxOutput = [np.argmax(_) for _ in output.data.cpu().numpy()]
    pred[range(BATCH_SIZE), maxOutput] = 1
    accuracy = sum(pred == labels_val.cpu().numpy())/len(labels_val)
    accuracy = accuracy[0]
    print('old accuracy: {}'.format(accuracy))



    val_accuracies_step.append(accuracy)
    # Track losses
    loss_val = criterion(output, labels_val)
    val_losses_step.append(loss_val.item())

    # Calculate and track validation accuracy



    # Compute gradients and optimise
    loss.backward()
    optimizer.step()

# Plot loss
plt.figure()
plt.title('Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.plot(range(len(train_losses_step)), train_losses_step, label='train_step')
plt.plot(range(len(val_losses_step)), val_losses_step, label='val_step')
plt.plot(range(0, len(val_losses_smooth)*STATS_STEPS, STATS_STEPS),
         val_losses_smooth,
         label='validation')
plt.plot(range(0, len(train_losses_smooth)*STATS_STEPS, STATS_STEPS),
         train_losses_smooth,
         label='train')
plt.grid(True)
plt.legend()

# Plot accuracy
plt.figure()
plt.title('Accuracy')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
#plt.plot(range(len(train_accuracies_step)), train_accuracies_step, label='train')
#plt.plot(range(len(val_accuracies_step)), val_accuracies_step, label='val')
plt.plot(range(0, len(val_accuracies_smooth)*STATS_STEPS, STATS_STEPS),
         val_accuracies_smooth,
         label='validation')
plt.plot(range(0, len(train_accuracies_smooth)*STATS_STEPS, STATS_STEPS),
         train_accuracies_smooth,
         label='train')
plt.plot(range(len(train_accuracies_step)), train_accuracies_step, label='train_step')
plt.plot(range(len(val_accuracies_step)), val_accuracies_step, label='val_step')
plt.grid(True)
plt.legend()

# Plot accuracy compare test val_accuracies_simple
plt.figure()
plt.title('Compare accuracy methods')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.plot(range(0, len(val_accuracies_simple)*STATS_STEPS, STATS_STEPS),
         val_accuracies_simple,
         label='validation_simple')
plt.plot(range(0, len(val_accuracies_smooth)*STATS_STEPS, STATS_STEPS),
         val_accuracies_smooth,
         label='validation_smooth')
plt.plot(range(len(val_accuracies_step)), val_accuracies_step, label='val_step')
plt.plot(range(0, len(val_accuracies_new)*STATS_STEPS, STATS_STEPS),
         val_accuracies_new, label='val_new')
plt.grid(True)
plt.legend()

# ROC Training
optimizer.zero_grad()
fpr, tpr, auc= get_roc_curve(
    net,
    dataset.get_batch_train_all(),
    dataset.get_labels_train_all(),
    optimizer,
    10)
plt.figure()
plt.title('ROC Curve - Training')
plt.plot(fpr, tpr, label='Area = {:.2f}'.format(auc))
plt.plot([0,1],[0,1], linestyle='dashed', color='k')
plt.grid(True)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend()
print('fpr:\n', fpr)
print('tpr:\n', tpr)

# ROC Validation
optimizer.zero_grad()
fpr, tpr, auc= get_roc_curve(
    net,
    dataset.get_batch_val_all(),
    dataset.get_labels_val_all(),
    optimizer,
    10)
plt.figure()
plt.title('ROC Curve - Validation')
plt.plot(fpr, tpr, label='Area = {:.2f}'.format(auc))
plt.plot([0,1],[0,1], linestyle='dashed', color='k')
plt.grid(True)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend()
print('fpr:\n', fpr)
print('tpr:\n', tpr)


params = list(net.parameters())
print(len(params))
for _ in params:
    print(_.size())
#print(params[0].size())
#print(net)

print('Running time:', '{:.2f}'.format(time.time() - start_time), ' s')

plt.show()
