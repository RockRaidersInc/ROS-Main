from __future__ import print_function

import torch
from torch import nn
import PIL

from nn_utils import *


class SmallerNet(torch.nn.Module):
    """
    This network appears to generalize quite well, even when only training with the small igvc_sim_trainset_3 dataset
    (test error is within 3% of training error). It doesn't work super well though.
    """
    def __init__(self):
        super(SmallerNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 12, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(12, 16, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 1, 3, padding=1)

        self.pool = torch.nn.MaxPool2d(2, 2)
        self.nonlinear = torch.nn.functional.elu
        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh
        self.dropout = torch.nn.Dropout(0.4)  # drop 50% of the neurons
    
    def forward(self, x_raw):
        x_prepared = x_raw / 128 - 1
        x1 = self.pool(self.nonlinear(self.conv1(x_prepared)))
        x2 = self.pool(self.nonlinear(self.conv2(x1)))
        x3 = self.pool(self.nonlinear(self.conv3(x2)))
        x4 = self.nonlinear(self.conv4(x3))
        output = self.tanh(x4)  # using tanh to get a value between -1 and 1
        return output


class LargerNet(torch.nn.Module):
    def __init__(self):
        super(LargerNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.conv2_1 = torch.nn.Conv2d(8, 12, 3, padding=1)
        self.conv2_2 = torch.nn.Conv2d(12, 12, 1, padding=0)
        self.conv2_3 = torch.nn.Conv2d(12, 12, 3, padding=1)

        self.conv3_1 = torch.nn.Conv2d(12, 16, 3, padding=1)
        self.conv3_2 = torch.nn.Conv2d(16, 32, 1, padding=0)
        self.conv3_3 = torch.nn.Conv2d(32, 16, 3, padding=1)

        self.conv4_1 = torch.nn.Conv2d(16, 36, 3, padding=1)
        self.conv4_2 = torch.nn.Conv2d(36, 36, 1, padding=0)
        self.conv4_3 = torch.nn.Conv2d(36, 1, 3, padding=1)

        self.pool = torch.nn.MaxPool2d(2, 2)
        self.nonlinear = torch.nn.functional.elu
        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh
        self.dropout = torch.nn.Dropout(0.4)  # drop 50% of the neurons
    
    def forward(self, x_raw):
        x_prepared = x_raw / 128 - 1
        x1 = self.pool(self.nonlinear(self.conv1(x_prepared)))
        x2_1 = self.nonlinear(self.conv2_1(x1))
        x2_2 = self.nonlinear(self.conv2_2(x2_1))
        x2_3 = self.pool(self.nonlinear(self.conv2_3(x2_2)))
        x3_1 = self.nonlinear(self.conv3_1(x2_3))
        x3_2 = self.nonlinear(self.conv3_2(x3_1))
        x3_3 = self.pool(self.nonlinear(self.conv3_3(x3_2)))
        x4_1 = self.nonlinear(self.conv4_1(x3_3))
        x4_2 = self.nonlinear(self.conv4_2(x4_1))
        x4_3 = self.nonlinear(self.conv4_3(x4_2))
        output = self.tanh(x4_3)  # using tanh to get a value between -1 and 1
        return output


class Yolo2Transfer(torch.nn.Module):
    def __init__(self, pretrained_weights):
        super(Yolo2Transfer, self).__init__()
        self.models = torch.nn.ModuleList()

        # [convolutional]
        # batch_normalize=1
        # filters=32
        # size=3
        # stride=1
        # pad=1
        # activation=leaky
        model = nn.Sequential()
        model.add_module('conv1', nn.Conv2d(3, 32, 3, 1, 1, bias=False))
        model.add_module('bn1', nn.BatchNorm2d(32))
        model.add_module('leaky1', nn.LeakyReLU(0.1, inplace=True))
        model.requires_grad = False
        self.models.append(model)

        # [maxpool]
        # size=2
        # stride=2
        model = nn.MaxPool2d(2, 2)
        self.models.append(model)

        # [convolutional]
        # batch_normalize=1
        # filters=64
        # size=3
        # stride=1
        # pad=1
        # activation=leaky
        model = nn.Sequential()
        model.add_module('conv2', nn.Conv2d(32, 64, 3, 1, 1, bias=False))
        model.add_module('bn2', nn.BatchNorm2d(64))
        model.add_module('leaky2', nn.LeakyReLU(0.1, inplace=True))
        model.requires_grad = False
        self.models.append(model)

        # [maxpool]
        # size=2
        # stride=2
        model = nn.MaxPool2d(2, 2)
        self.models.append(model)

        # [convolutional]
        # batch_normalize=1
        # filters=128
        # size=3
        # stride=1
        # pad=1
        # activation=leaky
        model = nn.Sequential()
        model.add_module('conv3', nn.Conv2d(64, 128, 3, 1, 1, bias=False))
        model.add_module('bn3', nn.BatchNorm2d(128))
        model.add_module('leaky3', nn.LeakyReLU(0.1, inplace=True))
        model.requires_grad = False
        self.models.append(model)

        # [convolutional]
        # batch_normalize=1
        # filters=64
        # size=1
        # stride=1
        # pad=1
        # activation=leaky
        model = nn.Sequential()
        model.add_module('conv4', nn.Conv2d(128, 64, 1, 1, 0, bias=False))
        model.add_module('bn4', nn.BatchNorm2d(64))
        model.add_module('leaky4', nn.LeakyReLU(0.1, inplace=True))
        model.requires_grad = False
        self.models.append(model)

        # [convolutional]
        # batch_normalize=1
        # filters=128
        # size=3
        # stride=1
        # pad=1
        # activation=leaky
        model = nn.Sequential()
        model.add_module('conv5', nn.Conv2d(64, 128, 3, 1, 1, bias=False))
        model.add_module('bn5', nn.BatchNorm2d(128))
        model.add_module('leaky5', nn.LeakyReLU(0.1, inplace=True))
        model.requires_grad = False
        self.models.append(model)

        # [maxpool]
        # size=2
        # stride=2
        model = nn.MaxPool2d(2, 2)
        self.models.append(model)

        # [convolutional]
        # batch_normalize=1
        # filters=256
        # size=3
        # stride=1
        # pad=1
        # activation=leaky
        model = nn.Sequential()
        model.add_module('conv6', nn.Conv2d(128, 256, 3, 1, 1, bias=False))
        model.add_module('bn6', nn.BatchNorm2d(256))
        model.add_module('leaky6', nn.LeakyReLU(0.1, inplace=True))
        model.requires_grad = False
        self.models.append(model)

        # [convolutional]
        # batch_normalize=1
        # filters=128
        # size=1
        # stride=1
        # pad=1
        # activation=leaky
        model = nn.Sequential()
        model.add_module('conv7', nn.Conv2d(256, 128, 1, 1, 0, bias=False))
        model.add_module('bn7', nn.BatchNorm2d(128))
        model.add_module('leaky7', nn.LeakyReLU(0.1, inplace=True))
        model.requires_grad = False
        self.models.append(model)

        # [convolutional]
        # batch_normalize=1
        # filters=256
        # size=3
        # stride=1
        # pad=1
        # activation=leaky
        model = nn.Sequential()
        model.add_module('conv8', nn.Conv2d(128, 256, 3, 1, 1, bias=False))
        model.add_module('bn8', nn.BatchNorm2d(256))
        model.add_module('leaky8', nn.LeakyReLU(0.1, inplace=True))
        model.requires_grad = False
        self.models.append(model)

        self.load_state_dict(torch.load("yolo2_partial_weights"))

        self.condense_1 = nn.Sequential()
        self.condense_1.add_module('condense_conv1', nn.Conv2d(256, 16, 3, 1, 1, bias=False))
        self.condense_1.add_module('condense_leaky1', nn.LeakyReLU(0.1, inplace=False))
        self.condense_1.requires_grad = True

        self.classify_module = model = nn.Sequential()
        self.classify_module.add_module('classifier_conv1', nn.Conv2d(64 + 16, 16, 1, 1, 0, bias=False))
        self.classify_module.add_module('classifier_leaky1', nn.LeakyReLU(0.1, inplace=False))
        self.classify_module.add_module('classifier_conv2', nn.Conv2d(16, 1, 1, 1, 0, bias=False))
        self.classify_module.add_module('classifier_leaky2', nn.Hardtanh())
        self.classify_module.requires_grad = True
    
    def forward(self, imgs):
        width = 416
        height = 416
        xs = []
        for img in imgs:
            img = PIL.Image.fromarray(np.array(img))
            sized = img.resize((width, width))
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(sized.tobytes()))
            img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
            img = img.view(3, 416, 416)
            xs.append(img.float().div(255.0))

        x = torch.stack(xs).to(get_default_torch_device())

        for i, model in enumerate(self.models):
            x = model(x)
            if i == 2:
                downsampled_4x = x
        
        # x is the output from the pretrained network at this point

        condensed = self.condense_1(x)
        upsampled = nn.functional.interpolate(condensed, size=(condensed.data.shape[2] * 4, condensed.data.shape[3] * 4))
        skip_link_concatenated = torch.cat((downsampled_4x, upsampled), dim=1)  # stack along the pixel value dimension

        classified = self.classify_module(skip_link_concatenated)

        return classified
