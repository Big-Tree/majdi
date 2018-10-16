import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # gradient descent
import random
import pydicom
import time
import numpy as np
import sys
sys.path.append('/vol/research/mammo2/will/python/usefulFunctions')
import usefulFunctions as uf


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
                '/vol/research/mammo2/will/data/prem/Segments',
                '2D_dim2d.dcm'),
            'lesions': uf.get_files(
                '/vol/research/mammo2/will/data/prem/2D/6mm',
                '*.dcm')}

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

        # Split the data into training and test
        # Mix the lesions and backgrounds
        # [[img, label], [img, label]]
        dataset_mixer = []
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
        self.test = {
            'data': np.asarray([_['image'] for _ in dataset_mixer][s_p : ]),
            'labels': np.asarray([_['class'] for _ in dataset_mixer][s_p : ])}
        # Convert to one hot labels
        tmp_labels = self.train['labels']
        self.train['labels'] = np.zeros((len(tmp_labels), 2))
        self.train['labels'][range(len(tmp_labels)), tmp_labels] = 1

        tmp_labels = self.test['labels']
        self.test['labels'] = np.zeros((len(tmp_labels), 2))
        self.test['labels'][range(len(tmp_labels)), tmp_labels] = 1

        # Reshape the images
        tmp = self.train['data'].shape
        self.train['data'].shape = (tmp[0], 1, tmp[1], tmp[2])
        tmp = self.test['data'].shape
        self.test['data'].shape = (tmp[0], 1, tmp[1], tmp[2])

        # Normalise between -1 and 1
        # Get max value
        max_pixel = np.amax([np.amax(self.train['data']),
                             np.amax(self.test['data'])])
        print('Max pixel: ', max_pixel)
        self.train['data'] = self.train['data']/max_pixel*2 - 1
        self.test['data'] = self.test['data']/max_pixel*2 - 1
        print('min self.train: ', np.amin(self.train['data']))
        print('max self.train: ', np.amax(self.train['data']))
        print('min self.test: ', np.amin(self.test['data']))
        print('max self.test: ', np.amax(self.test['data']))


    def get_batch_train(self):
        self.batch_number += 1
        out = self.train['data'][(self.batch_number-1)*self.batch_size :
            self.batch_number*self.batch_size]
        #out = out.astype(np.float64)
        print('out.shape: ', out.shape)
        print('out.dtype: ', out.dtype)
        out = torch.from_numpy(out).float()
        out = out.to(device)
        print('out.shape_post: ', out.shape)
        return out
    def get_labels_train(self):
        out = self.train['labels'][(self.batch_number-1)*self.batch_size :
            self.batch_number*self.batch_size]
        out = torch.from_numpy(out).float()
        out = out.to(device)
        print('out.dtype: ', out.dtype)
        return out


# Reproducing Majdi's work with his network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
        print('forward, x.type: ', x.type())
        print('0: ', x.shape)
        x = self.conv1(x)
        print('1: ', x.shape)
        x = F.relu(x)
        print('2: ', x.shape)
        x = F.max_pool2d(x, kernel_size=2, padding=1)
        print('3: ', x.shape)
        #x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, padding=1)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, padding=1)
        print('4: ', x.shape)
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=2, padding=1)
        x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=2, padding=1)
        print('5: ', x.shape)
        x = F.max_pool2d(F.relu(self.conv5(x)), kernel_size=2, padding=1)

        print('6: ', x.shape)
        x = x.view(-1, self.num_flat_features(x))
        print('7: ', x.shape)
        x = self.fc1(x)
        print('8: ', x.shape)
        x = F.softmax(x, dim=0) # Should it be put through a relu? Is it dim 0?
        print('9: ', x.shape)
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
device = torch.device('cpu')
device = torch.device('cuda:0') # Run on GPU
# Globals:
BATCH_SIZE = 25
STEPS = 20
DEVICE = torch.device('cuda:0')

dataset = LoadDataSet(0.9, BATCH_SIZE, DEVICE)
print('dataset.train[data].shape: ', dataset.train['data'].shape)
print('dataset.test[data].shape: ', dataset.test['data'].shape)
print('dataset.train[labels].shape: ', dataset.train['labels'].shape)
print('dataset.test[labels].shape: ', dataset.test['labels'].shape)
net = Net()
net = net.to(DEVICE) # Enable GPU
input = torch.randn(1, 1, 210, 210, device = DEVICE)
input = torch.randn(1, 1, 211, 211, device = DEVICE)
input = torch.randn(1, 1, 429, 429, device = DEVICE)
output = net(input)


# Use the gradients to update the weights
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Print the parameters before and after
# Before:
#print('Network params before')
#print(list(net.parameters()))
criterion = nn.MSELoss()
for i in range(STEPS):
    batch = dataset.get_batch_train()
#    batch = batch.float()
    optimizer.zero_grad()
    print('batch.shape: ', batch.shape)
    print('batch.type(): ', batch.type())
    output = net(batch)
    labels = dataset.get_labels_train()
#    labels = labels.float()
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()




# After:
#print(list(net.parameters()))

#print('\n\n\n')


params = list(net.parameters())
print(len(params))
for _ in params:
    print(_.size())
#print(params[0].size())
#print(net)

print('Running time:', '{:.2f}'.format(time.time() - start_time), ' s')
