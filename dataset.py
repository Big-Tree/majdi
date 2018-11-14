import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pydicom
import random
import sys
sys.path.append('/vol/research/mammo/mammo2/will/python/usefulFunctions')
import usefulFunctions as uf

# Class to read the dicom images in for training and testing of the network
# Function to read in the DICOM images, convert to numpy arrays, label and then
# loads the images into three classes for train, val and test.  These classes,
# which are returned will be used in the network main loop.
# A split ratio of 0.8 will set 80% of the images for training, 10% for
# validaiton and 10% for test
# Class 0 - backgrounds... Class 1 - lesions
def load_data_set(split_ratio, device, seed):
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
    if seed != None:
        random.seed(seed) # Fix datasets
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

def print_samples(dataloader, block, num_rows, num_cols):
    # Print some of the images
    fig = plt.figure()
    plt.ion()
    for i in range(num_cols*num_rows):
        plt.subplot(num_rows, num_cols, i+1)
        sample = next(iter(dataloader))
        label = sample['label'].numpy()[0]
        sample = sample['image']
        sample = np.asarray(sample.numpy())
        sample = sample[0,...]
        #sample = datasets['train'][0]
        img_stacked = np.stack((sample,)*3, axis=-1)
        img_stacked = np.squeeze(img_stacked)
        img_stacked = (img_stacked + 1)/2
        plt.imshow(img_stacked)
        plt.title(str(label))
    plt.show(block=block)
    #plt.pause(0.001) # Displays the figures I think

class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # For some reason in the tutoral they swap the calour channel to front
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}

# we need 3 datasets:
    #train
    #validation
    #test
# base class will return three dataset objects
# in the init the images will be passed

class MajdiDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        sample = {'image': self.images[idx], 'label': self.labels[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample
