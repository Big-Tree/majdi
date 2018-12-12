import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class oldVgg19Net(nn.Module):
    def __init__(self):
        #net = models.vgg19(pretrained=True)
        net = models.resnet18(pretrained=True)
        # Freeze model
        for param in net.parameters():
            param.requires_grad = False
        # Add head
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 2)

def vgg19Net():
    model = models.vgg19(pretrained=True)
    # add conv so that images with a single channel will work
    first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1,
                                  dilation=1, groups=1, bias=True)]
    first_conv_layer.extend(list(model.features))
    model.features= nn.Sequential(*first_conv_layer )

    # Freeze model
    for param in model.features.parameters():
        param.requires_grad = False
            param.requires_grad))

    # Newly created modules have require_grad=True by default
    #num_features = model.classifier[0].in_features
    num_features = 86528
    num_layers = len(list(model.classifier.children()))
    # Remove all layers
    features = list(
        model.classifier.children())[:-num_layers]
    # Freeze classifier
    for param in model.classifier.parameters():
        pass
        #param.requires_grad = False
    features.extend([nn.Linear(num_features, 2),
                    nn.Softmax(dim=1)]) # Add our layer (grads=true)
    #features.extend(F.softmax(x, dim=1)
    model.classifier = nn.Sequential(*features) # Replace the model classifier
    model.classifier = nn.Sequential(*[nn.Linear(num_features, 2),
                                      nn.Softmax(dim=1)])
    print(model)

    return model


def vgg19NetKaggle():
    # Load in the vgg net with its default weights
    # Somehow change the input
    # redefine the final classifier
    # how does the features feed into the classifier
    # vggNet expects images of size 224X224X3

    model = models.vgg19(pretrained=True)
    # add conv so that images with a single channel will work
    first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1,
                                  dilation=1, groups=1, bias=True)]
    first_conv_layer.extend(list(model.features))
    model.features= nn.Sequential(*first_conv_layer )

    #model = models.resnet18(pretrained=True)
    # Freeze model
    for param in model.features.parameters():
        param.requires_grad = False
    print('model:\n{}'.format(model))

    # Newly created modules have require_grad=True by default
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1] # Remove last layer
    # Freeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = False
    features.extend([nn.Linear(num_features, 2), nn.Softmax(dim=1)]) # Add our layer (grads=true)
    model.classifier = nn.Sequential(*features) # Replace the model classifier
    print(model)

    return model



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


