import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
import pydicom
import random
from sklearn.preprocessing import normalize
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
            file_name = f.split('/')[-3]
            dicom_images[key].append({
                file_name: pydicom.dcmread(f)})
            if index % 10 == 0:
                print('    ', index, '/', len(file_list[key]), end='\r')
    rgb_images = {'backgrounds':[], 'lesions':[]}
    print('Converting dicom to RGB...')
    for key in rgb_images:
        for dic in dicom_images[key]:
            file_name = [_ for _ in dic][0]
            rgb_images[key].append({
                file_name: img.pixel_array})

    # Split the data into training and val
    # Mix the lesions and backgrounds
    # [[img, label], [img, label]]
    # Note - dictionaries preserve insertion order
    dataset_mixer = []
    it = iter(rgb_images)
    print('class 0: {}'.format(next(it)))
    print('class 1: {}'.format(next(it)))
    for index, key in enumerate(rgb_images):
        for dic in rgb_images[key]:
            dataset_mixer.append({
                'image': dic,
                'class': index})
    if seed != None:
        random.seed(seed) # Fix datasets
    random.shuffle(dataset_mixer)
    # Set split points for train, val, test
    s_p = round(split_ratio*len(dataset_mixer))
    e_p = len(dataset_mixer)
    s_p_val_test = round((s_p+e_p)/2) # DELETE
    s_p_left = [0, s_p, round((s_p+e_p)/2)]
    s_p_right = [s_p, round((s_p+e_p)/2), None]
    datasets = {'train': None,
               'val': None,
               'test': None}
    for i, key in enumerate(datasets):
        for file_name in key:
            # Split into train, val, test
            datasets[key] = {
                'data': np.asarray(
                    [_['image'] for _ in \
                        dataset_mixer][s_p_left[i]:s_p_right[i]],
                dtype=np.float32),
                'labels': np.asarray(
                    [_['class'] for _ in \
                        dataset_mixer][s_p_left[i]:s_p_right[i]])}
            # Convert to one hot labels
            tmp_labels = datasets[key]['labels']
            datasets[key]['labels'] = np.zeros((len(tmp_labels), 2))
            datasets[key]['labels'][range(len(tmp_labels)), tmp_labels] = 1
            # Reshape the images
            tmp = datasets[key]['data'][file_name].shape
            #tmp = [n, H, W]
            # torchvision.transforms requires [n, H, W, C]
            datasets[key]['data'][file_name].shape = (
                tmp[0], 1, tmp[1], tmp[2])
            print('datasets[{}][data][file_name].shape{}'.format(
                key,datasets[key]['data'][file_name].shape))

    # Get max value
    #image_array = [list(_.values())[0] for _ in datasets[key]
    max_pixel = np.amax([np.amax(datasets[key]['data']) for key in datasets])
    print('Max pixel: ', max_pixel)
    # Normalise between 0 and 1
    for key in datasets:
        datasets[key]['data'] = datasets[key]['data']/max_pixel*1 -0 #*2 - 1
        print('min {}: {}'.format(key, np.amin(datasets[key]['data'])))
        print('max {}: {}'.format(key, np.amax(datasets[key]['data'])))

    # Load the images into the dataset class
    out = {'train':None,
           'val':None,
           'test':None}
   # data_transforms = {
   #     'train': transforms.Compose([
   #         transforms.ToPILImage(),
   #         transforms.RandomRotation(360),
   #         PILToTensor()]),
   #     'val': None,
   #     'test': None}
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(360),
            PILToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224,  0.225]),
            NoTriangles()]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            PILToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224,  0.225]),
            NoTriangles()]),
        'test':transforms.Compose([
            transforms.ToPILImage(),
            PILToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224,  0.225]),
            NoTriangles()])}
    for key in out:
        out[key] = MajdiDataset(datasets[key]['data'],
                             datasets[key]['labels'],
                             transform = data_transforms[key])
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
        # Revert form -1 to 1
        #img_stacked = (img_stacked + 1)/2
        img_stacked = ((img_stacked - img_stacked.min())
                       / (img_stacked.max() - img_stacked.min()))
        plt.imshow(img_stacked)
        plt.title(str(label))
    plt.show(block=block)
    #plt.pause(0.001) # Displays the figures I think


# The transforms.ToTensor() messes with the channel ordering and pixel range
# transforms.toPILImage also removes the single channel ffs!
# PIL format [x,y,c]
# Torch format [c,x,y]
class PILToTensor():
    def __call__(self, sample):
        # Convert to numpy
        to_numpy = np.array(sample)
        # Add back in the channel dimesion
        # Annoyingly the transforms removed it
        add_channel = to_numpy
        add_channel.shape = (1, to_numpy.shape[0], to_numpy.shape[1])
        # Convert to tensor
        to_tensor = torch.from_numpy(to_numpy)
        return to_tensor

class NumpyToTensor():
    def __call__(self, x):
        return torch.from_numpy(x)


# Simple crops the image
# Can be used after a rotation to removed the dreaded triangles
# Should work for both numpy and tensors
class NoTriangles():
    def __call__(self, x):
        # 0.5^0.5 is the fraction at which we need to reduce the sides
        # width == height
        length = x.shape[1]
        lower = round(length*(1 - 0.5**0.5)/2)
        upper = round(length * (1 - (1-0.5**0.5)/2))
        x = x[:, lower:upper, lower:upper]
        return x


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
            # Perform transforms on images
            # Swap channel order for PIL
            sample['image'].shape = (sample['image'].shape[1],
                                     sample['image'].shape[2], 1)
            sample['image'] = self.transform(sample['image'])
            # Convert labels to tensor (image already converted to tensor)
            sample['label'] = torch.from_numpy(sample['label'])
        else:
            # If no transforms we still need to convert to tensor
            sample['image'] = torch.from_numpy(sample['image'])
            sample['label'] = torch.from_numpy(sample['label'])


        return sample
