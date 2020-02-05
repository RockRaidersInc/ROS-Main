from __future__ import print_function
import time

import torch
from torch import nn
import PIL

from nn_utils import *


class GrassNet(torch.nn.Module):
    """
    This network is meant to only tell grass from not grass (where lines painted on grass count as grass). 
    We'll see how well it works at detecting lanes as grass
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


class Yolo2Base(torch.nn.Module):
    def __init__(self, pretrained_weights=None):
        super(Yolo2Base, self).__init__()

        self.model_name_index = {}
        
        self.width = 416
        self.height = 416
        
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
        self.model_name_index[len(self.model_name_index)] = "conv1"

        # [maxpool]
        # size=2
        # stride=2
        model = nn.MaxPool2d(2, 2)
        self.models.append(model)
        self.model_name_index[len(self.model_name_index)] = "max_pool_1"

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
        self.model_name_index[len(self.model_name_index)] = "conv2"

        # [maxpool]
        # size=2
        # stride=2
        model = nn.MaxPool2d(2, 2)
        self.models.append(model)
        self.model_name_index[len(self.model_name_index)] = "max_pool_2"

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
        self.model_name_index[len(self.model_name_index)] = "conv3"

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
        self.model_name_index[len(self.model_name_index)] = "conv4"

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
        self.model_name_index[len(self.model_name_index)] = "conv5"

        # [maxpool]
        # size=2
        # stride=2
        model = nn.MaxPool2d(2, 2)
        self.models.append(model)
        self.model_name_index[len(self.model_name_index)] = "max_pool_3"

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
        self.model_name_index[len(self.model_name_index)] = "conv6"

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
        self.model_name_index[len(self.model_name_index)] = "conv7"

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
        self.model_name_index[len(self.model_name_index)] = "conv8"

        if pretrained_weights is None:
            self.load_state_dict(torch.load("yolo2_partial_weights"))
    

    def prep_images(self, imgs):
        """
        Rescale, re-order axes, and move images to GPU (if used) to prepare input images for the neural network
        """
        # width = 416
        # height = 416
        xs = []
        for img in imgs:
            # img = PIL.Image.fromarray(np.array(img))
            # sized = img.resize((width, width))
            # sized = img
            img_buffer = torch.ByteTensor(torch.ByteStorage.from_buffer(img))
            img = img_buffer.view(img.shape[0], img.shape[1], 3).transpose(0,1).transpose(0,2).contiguous()
            # img = img.view(3, height, width)
            xs.append(img.float().div(255.0))

        x = torch.stack(xs).to(get_default_torch_device())
        return x



class Yolo2Transfer(Yolo2Base):
    def __init__(self, pretrained_weights=None):
        super(Yolo2Transfer, self).__init__(pretrained_weights=pretrained_weights)

        self.downsample_2x_condense = nn.Sequential()
        self.downsample_2x_condense.add_module('condense_downsample_2x_conv1', nn.Conv2d(32, 16, 1, 1, 0, bias=True))
        self.downsample_2x_condense.add_module('condense_downsample_2x_leaky', nn.LeakyReLU(0.1, inplace=False))

        self.downsample_2x_condense.requires_grad = True
        self.downsample_4x_condense = nn.Sequential()
        self.downsample_4x_condense.add_module('condense_downsample_4x_conv1', nn.Conv2d(64, 128, 3, 1, 0, bias=True))
        self.downsample_4x_condense.add_module('condense_downsample_4x_leaky1', nn.LeakyReLU(0.1, inplace=False))
        self.downsample_4x_condense.add_module('condense_downsample_4x_conv2', nn.Conv2d(128, 32, 1, 1, 0, bias=True))
        self.downsample_4x_condense.add_module('condense_downsample_4x_leaky2', nn.LeakyReLU(0.1, inplace=False))
        self.downsample_4x_condense.requires_grad = True

        self.downsample_8x_condense = nn.Sequential()
        self.downsample_8x_condense.add_module('condense_downsample_8x_conv1', nn.Conv2d(128, 128, 3, 1, 0, bias=True))
        self.downsample_8x_condense.add_module('condense_downsample_8x_leaky1', nn.LeakyReLU(0.1, inplace=False))
        self.downsample_8x_condense.add_module('condense_downsample_8x_conv2', nn.Conv2d(128, 64, 1, 1, 0, bias=True))
        self.downsample_8x_condense.add_module('condense_downsample_8x_leaky3', nn.LeakyReLU(0.1, inplace=False))
        self.downsample_8x_condense.requires_grad = True

        self.downsample_8x_2_condense = nn.Sequential()
        self.downsample_8x_2_condense.add_module('condense_downsample_8x_2_conv1', nn.Conv2d(256, 256, 3, 1, 0, bias=True))
        self.downsample_8x_2_condense.add_module('condense_downsample_8x_2_leaky1', nn.LeakyReLU(0.1, inplace=False))
        self.downsample_8x_2_condense.add_module('condense_downsample_8x_2_conv2', nn.Conv2d(256, 64, 1, 1, 0, bias=True))
        self.downsample_8x_2_condense.add_module('condense_downsample_8x_2_leaky2', nn.LeakyReLU(0.1, inplace=False))
        self.downsample_8x_2_condense.requires_grad = True

        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier_conv1', nn.Conv2d(16 + 32 + 64 + 64, 128, 3, stride=1, padding=1, bias=True))
        self.classifier.add_module('classifier_leaky1', nn.LeakyReLU(0.1, inplace=False))
        self.classifier.add_module('classifier_conv2', nn.Conv2d(128, 64, 1, 1, padding=0, bias=True))
        self.classifier.add_module('classifier_leaky2', nn.LeakyReLU(0.1, inplace=False))
        self.classifier.add_module('classifier_conv3', nn.Conv2d(64, 128, 3, 1, padding=1, bias=True))
        self.classifier.add_module('classifier_leaky3', nn.LeakyReLU(0.1, inplace=False))
        self.classifier.add_module('classifier_conv4', nn.Conv2d(128, 64, 1, stride=1, padding=0, bias=True))
        self.classifier.add_module('classifier_leaky4', nn.LeakyReLU(0.1, inplace=False))
        self.classifier.add_module('classifier_conv5', nn.Conv2d(64, 1, 3, stride=1, padding=1, bias=True))
        self.classifier.requires_grad = True

        if pretrained_weights is not None:
            self.load_state_dict(torch.load(pretrained_weights))

    def forward(self, imgs):
        """
        Run the neural network. imgs should be a list of PIL images, all with the same height/width
        """
        x = self.prep_images(imgs)
        output_size = x.data.shape[2:4]

        for i, model in enumerate(self.models):
            x = model(x)
            if self.model_name_index[i] == "max_pool_1":
                downsampled_2x = x
            if self.model_name_index[i] == "max_pool_2":
                downsampled_4x = x
            elif self.model_name_index[i] == "max_pool_3":
                downsampled_8x = x
            elif self.model_name_index[i] == "conv8":
                downsampled_8x_2 = x
                break
        # x is the output from the pretrained network at this point

        condensed_2x = self.downsample_2x_condense(downsampled_2x)
        condensed_4x = self.downsample_4x_condense(downsampled_4x)
        condensed_8x = self.downsample_8x_condense(downsampled_8x)
        condensed_8x_2 = self.downsample_8x_2_condense(downsampled_8x_2)

        classification_size = condensed_4x.data.shape[2:4]

        resized_2x = nn.functional.interpolate(condensed_2x, size=classification_size)
        resized_4x = nn.functional.interpolate(condensed_4x, size=classification_size)
        resized_8x = nn.functional.interpolate(condensed_8x, size=classification_size)
        resized_8x_2 = nn.functional.interpolate(condensed_8x_2, size=classification_size)
        skip_link_concatenated = torch.cat((resized_2x, resized_4x, resized_8x, resized_8x_2), dim=1)  # stack along the pixel value dimension

        classified = self.classifier(skip_link_concatenated)
        
        original_size = nn.functional.interpolate(classified, size=output_size)  # resize the output back to the input's origional size
        return original_size


class Yolo2Transfer_smaller(Yolo2Base):
    """
    The goal of this network is to be smller, using fewer layers from YOLO2 than the above network 
    """
    def __init__(self, pretrained_weights=None):
        super(Yolo2Transfer_smaller, self).__init__(pretrained_weights=pretrained_weights)

        self.condense_2 = nn.Sequential()
        self.condense_2.add_module('condense_2_conv1', nn.Conv2d(64, 8, 1, 1, 0, bias=True))
        self.condense_2.add_module('condense_2_leaky1', nn.LeakyReLU(0.1, inplace=False))
        self.condense_2.requires_grad = True

        self.condense_3 = nn.Sequential()
        self.condense_3.add_module('condense_3_conv1', nn.Conv2d(64, 3, 1, 1, 0, bias=True))
        self.condense_3.add_module('condense_3_leaky1', nn.LeakyReLU(0.1, inplace=False))
        self.condense_3.requires_grad = True

        self.classify_module = model = nn.Sequential()
        self.classify_module.add_module('classifier_conv1', nn.Conv2d(64, 128, 3, 1, 0, bias=True))
        self.classify_module.add_module('classifier_leaky1', nn.LeakyReLU(0.1, inplace=False))
        self.classify_module.add_module('classifier_conv3', nn.Conv2d(128, 64, 1, 1, 0, bias=True))
        self.classify_module.add_module('classifier_leaky3', nn.LeakyReLU(0.1, inplace=False))
        self.classify_module.add_module('classifier_conv4', nn.Conv2d(64, 128, 3, 1, 0, bias=True))
        self.classify_module.add_module('classifier_leaky4', nn.LeakyReLU(0.1, inplace=False))
        self.classify_module.add_module('classifier_conv5', nn.Conv2d(128, 64, 1, 1, 0, bias=True))
        self.classify_module.add_module('classifier_leaky5', nn.LeakyReLU(0.1, inplace=False))
        self.classify_module.add_module('classifier_conv6', nn.Conv2d(64, 128, 3, 1, 0, bias=True))
        self.classify_module.add_module('classifier_leaky6', nn.LeakyReLU(0.1, inplace=False))
        self.classify_module.add_module('classifier_conv7', nn.Conv2d(128, 1, 1, 1, 0, bias=True))

        self.classify_module.requires_grad = True

        if pretrained_weights is not None:
            self.load_state_dict(torch.load(pretrained_weights))

    def forward(self, imgs):
        """
        Run the neural network. imgs should be a list of PIL images, all with the same height/width
        """
        x = self.prep_images(imgs)

        for i, model in enumerate(self.models):
            x = model(x)
            if self.model_name_index[i] == "max_pool_2":
                downsampled_4x = x
                break
            # if self.model_name_index[i] == "conv4":
            #     downsampled_8x = x
            #     break  # this is the last layer we want
        
        # condensed_lower = self.condense_2(downsampled_8x)
        # upsampled_lower = nn.functional.interpolate(condensed_lower, size=(condensed_lower.data.shape[2] * 2, condensed_lower.data.shape[3] * 2))
        # condensed_4x = self.condense_3(downsampled_4x)
        # skip_link_concatenated = torch.cat((condensed_4x, upsampled_lower), dim=1)  # stack along the pixel value dimension

        classified = self.classify_module(x)
        
        original_size = nn.functional.interpolate(classified, size=np.array(imgs[0]).shape[:2])  # resize the output back to the input's origional size
        return original_size

