import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
import pydicom
import random
from sklearn.preprocessing import normalize
from tqdm import tqdm
import pickle
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
def load_data_set(split_ratio, device, seed, i_split=0):
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
        for f in tqdm(file_list[key], ascii=True):
        #for index, f in enumerate(file_list[key]):
            file_name = f.split('/')[-3]
            dicom_images[key].append({
                file_name: pydicom.dcmread(f)})
            #if index % 10 == 0:
                #print('    ', index, '/', len(file_list[key]), end='\r')
    rgb_images = {'backgrounds':[], 'lesions':[]}
    print('Converting dicom to RGB...')
    for key in rgb_images:
        for dic in dicom_images[key]:
            for file_name in dic:
                rgb_images[key].append({
                    file_name: np.asarray(
                        dic[file_name].pixel_array, dtype=np.float32)})

    # Normalise the images between 0 and 1
    background_images = [list(_.values())[0] for _ in
                         rgb_images['backgrounds']]
    lesion_images = [list(_.values())[0] for _ in rgb_images['lesions']]
    all_images = np.concatenate((background_images, lesion_images))
    print('all_images.shape: {}'.format(all_images.shape))
    max_pixel = np.amax(all_images)
    print('max_pixel: {}'.format(max_pixel))
    for key in rgb_images:
        for dic in rgb_images[key]:
            for file_name in dic:
                dic[file_name] = dic[file_name] / max_pixel
                #Reshape image
                tmp_shape = dic[file_name].shape
                dic[file_name].shape = (1,
                                        tmp_shape[0],
                                        tmp_shape[1])
                # torchvision.transforms requires [n, H, W, C]
                # pytorch chanel order: [n, C, H, W]


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
    random.seed(None)
    # Set split points for train, val, test
    s_p = round(split_ratio*len(dataset_mixer))
    e_p = len(dataset_mixer)
#---------------------------------------------------------------------------
    # i_split will increment by the number of images in test
    #i_split = (round(split_ratio*len(dataset_mixer)/2)) * i_split # DELETE
    i_split = (round((1-split_ratio)/2 * i_split * len(dataset_mixer)))
    split_point = {'train': np.arange(s_p) + i_split,
                   'val': np.arange(s_p, round((s_p+e_p)/2)) + i_split,
                   'test': np.arange(round((s_p+e_p)/2), e_p) + i_split}
    print('split_point: {}'.format(split_point))
    print('i_split: {}'.format(i_split))

    datasets = {'train': None,
               'val': None,
               'test': None}
    tmp_images = np.asarray([_['image'] for _ in dataset_mixer])
    tmp_classes = np.asarray([_['class'] for _ in dataset_mixer])
    print('tmp_images.shape: {}'.format(tmp_images.shape))
    print('tmp_classes.shape: {}'.format(tmp_classes.shape))
    for key in datasets:
        datasets[key] = {
            'data': np.take(tmp_images,
                split_point[key],
                mode='wrap'),
            'labels': np.take(tmp_classes,
                split_point[key],
                mode='wrap')}

        # Convert to one hot labels
        tmp_labels = datasets[key]['labels']
        datasets[key]['labels'] = np.zeros((len(tmp_labels), 2))
        datasets[key]['labels'][range(len(tmp_labels)), tmp_labels] = 1


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


def load_data_set_pickle(split_ratio, device, seed, i_split=0):
    # Initialise class variables:
    # Load in the dicom files
    # Get file list
    print('Getting file list...')
    # Load in the images from the pickle
    # Get location of all the lesion and contrilateral pickle files
    file_list = {
        'backgrounds': uf.get_files(
            '/vol/research/mammo/mammo2/will/data/batches/roi',
            'lesions.pickle'),
        'lesions': uf.get_files(
            '/vol/research/mammo/mammo2/will/data/batches/roi',
            'contrilaterals.pickle')}
    # -------------------Balance the dataset
    # Load the pickles in to memory
    rgb_images = {'backgrounds':[],
                  'lesions':[]}
    for key in file_list:
        print('Loading {}...'.format(key))
        for f in tqdm(file_list[key], ascii=True):
            with open(f, 'rb') as path:
                loaded = pickle.load(path)
                if key == 'backgrounds':
                    for img_name in loaded:
                        #print('len(rgb_images[key]: {}'.format(
                        #    len(rgb_images[key])))
                        #print('loaded[img_name].shape: {}'.format(
                        #    loaded[img_name].shape))
                        if loaded[img_name].shape == (256,256,1):
                            rgb_images[key].append(loaded[img_name])
                        else:
                            print('***Shape change!***')
                            print('loaded[img_name.shape]: {}'.format(
                                loaded[img_name].shape))
                else:
                    for tmp_img in loaded:
                        if tmp_img.shape == (256, 256, 1):
                            rgb_images[key].append(tmp_img)
                        else:
                            print('***Shape change!***')
                            print('tmp_img.shape: {}'.format(
                                tmp_img.shape))
        rgb_images[key] = np.asarray(rgb_images[key], dtype=np.float32)
    # Balance classes
    # Theres more backgrounds
    rgb_images['backgrounds'] = rgb_images['backgrounds'][:len(
        rgb_images['lesions'])]

    print('len(rgb_images[key]): {}'.format(len(rgb_images[key])))
    print('type(rgb_images[key]: {}'.format(type(rgb_images[key])))
    #tmp = np.asarray(rgb_images[key])
    #tmp2 = np.asarray(rgb_images[key], dtype=np.float32)
    #rgb_images[key] = np.asarray(rgb_images[key], dtype=np.float32)
    first_lesion = rgb_images['lesions'][0]
    first_normal = rgb_images['backgrounds'][0]
    print('len(rgb_images["lesions"]): {}'.format(len(rgb_images['lesions'])))
    print('len(rgb_images["backgrounds"]): {}'.format(len(rgb_images['backgrounds'])))
    print('first_lesion.shape: {}'.format(first_lesion.shape))
    print('first_normal.shape: {}'.format(first_normal.shape))


    # Normalise the images between 0 and 1
    max_value = np.asarray(list(rgb_images.values())).max()
    for key in rgb_images:
        rgb_images[key] /= max_value

    # Reshape the images
        # torchvision.transforms requires [n, H, W, C]
        # pytorch chanel order: [n, C, H, W]
    for key in rgb_images:
        for img in rgb_images[key]:
            img.shape = (1, img.shape[0], img.shape[1])

   # background_images = [list(_.values())[0] for _ in
   #                      rgb_images['backgrounds']]
   # lesion_images = [list(_.values())[0] for _ in rgb_images['lesions']]
   # all_images = np.concatenate((background_images, lesion_images))
   # print('all_images.shape: {}'.format(all_images.shape))
   # max_pixel = np.amax(all_images)
   # print('max_pixel: {}'.format(max_pixel))
   # for key in rgb_images:
   #     for dic in rgb_images[key]:
   #         for file_name in dic:
   #             dic[file_name] = dic[file_name] / max_pixel
   #             #Reshape image
   #             tmp_shape = dic[file_name].shape
   #             dic[file_name].shape = (1,
   #                                     tmp_shape[0],
   #                                     tmp_shape[1])
   #             # torchvision.transforms requires [n, H, W, C]
   #             # pytorch chanel order: [n, C, H, W]


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
    random.seed(None)
    # Set split points for train, val, test
    s_p = round(split_ratio*len(dataset_mixer))
    e_p = len(dataset_mixer)
#---------------------------------------------------------------------------
    # i_split will increment by the number of images in test
    #i_split = (round(split_ratio*len(dataset_mixer)/2)) * i_split # DELETE
    i_split = (round((1-split_ratio)/2 * i_split * len(dataset_mixer)))
    split_point = {'train': np.arange(s_p) + i_split,
                   'val': np.arange(s_p, round((s_p+e_p)/2)) + i_split,
                   'test': np.arange(round((s_p+e_p)/2), e_p) + i_split}
    print('split_point: {}'.format(split_point))
    print('i_split: {}'.format(i_split))

    datasets = {'train': None,
               'val': None,
               'test': None}
    tmp_images = np.asarray([_['image'] for _ in dataset_mixer])
    tmp_classes = np.asarray([_['class'] for _ in dataset_mixer])
    print('tmp_images.shape: {}'.format(tmp_images.shape))
    print('tmp_classes.shape: {}'.format(tmp_classes.shape))
    for key in datasets:
        datasets[key] = {
            'data': np.take(tmp_images,
                split_point[key],
                mode='wrap'),
            'labels': np.take(tmp_classes,
                split_point[key],
                mode='wrap')}

        # Convert to one hot labels
        tmp_labels = datasets[key]['labels']
        datasets[key]['labels'] = np.zeros((len(tmp_labels), 2))
        datasets[key]['labels'][range(len(tmp_labels)), tmp_labels] = 1


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
        print('creating dataset')
        self.images = images
        self.labels = labels
        self.transform = transform
        print('self.images.shape: {}'.format(self.images.shape))
        print('self.labels.shape: {}'.format(self.labels.shape))
        file_name = list(self.images[0].keys())[0]
        print('dataset - image size: {}'.format(
            self.images[0][file_name].shape))
        file_name = list(self.images[0].keys())[0]
        self.pil_shape = (self.images[0][file_name].shape[1],
                          self.images[0][file_name].shape[2],
                          1)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        #sample = {'image': self.images[idx], 'label': self.labels[idx]}
        file_name = list(self.images[idx].keys())[0]
        sample = {'image': self.images[idx][file_name],
                  'label': self.labels[idx],
                  'file_name': file_name}

        if self.transform:
            # Perform transforms on images
            # torchvision.transforms requires [n, H, W, C] if from numpy
            # pytorch chanel order: [n, C, H, W]
            # Swap channel order for PIL
            new_shape = (sample['image'].shape[1],
                        sample['image'].shape[2],
                        1)
            #new_shape = (429,429,1)
            sample['image'].shape = self.pil_shape
            sample['image'] = self.transform(sample['image'])
            # Convert labels to tensor (image already converted to tensor)
            sample['label'] = torch.from_numpy(sample['label'])
        else:
            # If no transforms we still need to convert to tensor
            sample['image'] = torch.from_numpy(sample['image'])
            sample['label'] = torch.from_numpy(sample['label'])


        return sample
