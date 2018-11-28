import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class vgg19Net(nn.Module):
    def __init__(self)
        net = torchvision.models.vgg19(pretrained=True)
        # Freeze model
        for param in model_conv.parameters():
            param.requires_grad = False
        # Add head
        num_ftrs = model_conv.fc.in_features


# Reproducing Majdi's work with his network architecture
# init with an image to compute correct sizes
class MajdiNet(nn.Module):
    def __init__(self, image, verbose=False):
        super(MajdiNet, self).__init__()
        self.verbose = verbose
        self.save_softmax = False
        print('MajdiNet__init image.shape: {}'.format(image.shape))
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 32, 2, padding=1)
        x = self.conv1(image)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, padding=1)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, padding=1)
        self.conv3 = nn.Conv2d(64, 96, 2, padding=1)
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=2, padding=1)
        self.conv4 = nn.Conv2d(96, 128, 2, padding=1)
        x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        x = F.max_pool2d(F.relu(self.conv5(x)), kernel_size=2, padding=1)
        #self.fc1 = nn.Linear(256 * 15 * 15, 2)
        self.fc1 = nn.Linear(x.shape[1]*x.shape[2]*x.shape[3], 2)
        print('MajdiNet__init__: {}'.format(x.shape))

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


