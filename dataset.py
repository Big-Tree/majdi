import torch
from torch.utils.data import Dataset


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
