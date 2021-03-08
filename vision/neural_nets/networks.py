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
    def __init__(self, pretrained_weights=None, init_weights=True):
        super(Yolo2Base, self).__init__()

        self.model_name_index = {}
        
        self.width = 416
        self.height = 416
        
        self.models = torch.nn.ModuleList()

        track_running_stats = True
        bn_momentum = 1

        # [convolutional]
        # batch_normalize=1
        # filters=32
        # size=3
        # stride=1
        # pad=1
        # activation=leaky
        model = nn.Sequential()
        model.add_module('conv1', nn.Conv2d(3, 32, 3, 1, 1, bias=False))
        model.add_module('bn1', nn.BatchNorm2d(32, momentum=bn_momentum))
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
        model.add_module('bn2', nn.BatchNorm2d(64, momentum=bn_momentum))
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
        model.add_module('bn3', nn.BatchNorm2d(128, momentum=bn_momentum))
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
        model.add_module('bn4', nn.BatchNorm2d(64, momentum=bn_momentum))
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
        model.add_module('bn5', nn.BatchNorm2d(128, momentum=bn_momentum))
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
        model.add_module('bn6', nn.BatchNorm2d(256, momentum=bn_momentum))
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
        model.add_module('bn7', nn.BatchNorm2d(128, momentum=bn_momentum))
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
        model.add_module('bn8', nn.BatchNorm2d(256, momentum=bn_momentum))
        model.add_module('leaky8', nn.LeakyReLU(0.1, inplace=True))
        model.requires_grad = False
        self.models.append(model)
        self.model_name_index[len(self.model_name_index)] = "conv8"

        if pretrained_weights is None and init_weights:
            # state_dict = torch.load("yolo2_partial_weights")
            # print(state_dict["models.0.bn1.running_mean"])  #, "models.0.bn1.running_var", "models.0.bn1.num_batches_tracked"
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
            # img_buffer = torch.ByteTensor(torch.ByteStorage.from_buffer(img))
            img_buffer = torch.from_numpy(img).to(get_default_torch_device())

            img = img_buffer.view(img.shape[0], img.shape[1], 3).transpose(0,1).transpose(0,2).contiguous()
            # img = img.view(3, height, width)
            xs.append(img.float().div(255.0))

        x = torch.stack(xs)
        return x


class Yolo2Transfer(Yolo2Base):
    def __init__(self, pretrained_weights=None):
        super(Yolo2Transfer, self).__init__(pretrained_weights=pretrained_weights)

        self.downsample_2x_condense = nn.Sequential()
        self.downsample_2x_condense.add_module('condense_downsample_2x_conv1', nn.Conv2d(32, 16, 3, stride=1, padding=1, bias=True))
        self.downsample_2x_condense.add_module('condense_downsample_2x_leaky1', nn.LeakyReLU(0.1, inplace=False))


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
                break
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


class Yolo2TransferSmaller(Yolo2Base):
    """
    The goal of this network is to be smaller, using fewer layers than the normal yolo2 transfer network
    """
    def __init__(self, pretrained_weights=None):
        super(Yolo2TransferSmaller, self).__init__(pretrained_weights=pretrained_weights)

        self.downsample_2x_condense = nn.Sequential()
        self.downsample_2x_condense.add_module('condense_2x_conv1', nn.Conv2d(32, 8, 3, stride=1, padding=1, bias=True))
        self.downsample_2x_condense.add_module('condense_2x_leaky1', nn.LeakyReLU(0.1, inplace=False))
        self.downsample_2x_condense.add_module('maxpool_4x_1', nn.MaxPool2d(2, 2))
        self.downsample_2x_condense.add_module('condense_2x_conv2', nn.Conv2d(8, 4, 3, stride=1, padding=1, bias=True))
        self.downsample_2x_condense.add_module('condense_2x_leaky2', nn.LeakyReLU(0.1, inplace=False))
        self.downsample_2x_condense.requires_grad = True


        self.downsample_4x_condense = nn.Sequential()
        self.downsample_4x_condense.add_module('condense_4x_conv1', nn.Conv2d(64, 128, 3, stride=3, padding=1, dilation=1, bias=True))
        self.downsample_4x_condense.add_module('condense_4x_leaky1', nn.LeakyReLU(0.1, inplace=False))
        self.downsample_4x_condense.add_module('condense_4x_conv2', nn.Conv2d(128, 96, 1, stride=1, padding=0, bias=True))
        self.downsample_4x_condense.add_module('condense_4x_leaky2', nn.LeakyReLU(0.1, inplace=False))

        self.downsample_4x_condense.add_module('maxpool_4x_1', nn.MaxPool2d(2, 2))

        self.downsample_4x_condense.add_module('condense_4x_conv10', nn.Conv2d(96, 128, 3, stride=1, padding=1, dilation=1, bias=True))
        self.downsample_4x_condense.add_module('condense_4x_leaky10', nn.LeakyReLU(0.1, inplace=False))
        self.downsample_4x_condense.add_module('condense_4x_conv11', nn.Conv2d(128, 96, 1, stride=1, padding=0, bias=True))
        self.downsample_4x_condense.add_module('condense_4x_leaky11', nn.LeakyReLU(0.1, inplace=False))

        self.downsample_4x_condense.add_module('condense_4x_conv12', nn.Conv2d(96, 128, 3, stride=1, padding=2, dilation=2, bias=True))
        self.downsample_4x_condense.add_module('condense_4x_leaky12', nn.LeakyReLU(0.1, inplace=False))
        self.downsample_4x_condense.add_module('condense_4x_conv13', nn.Conv2d(128, 64, 3, stride=1, padding=2, dilation=2, bias=True))
        self.downsample_4x_condense.add_module('condense_4x_leaky13', nn.LeakyReLU(0.1, inplace=False))
        self.downsample_4x_condense.requires_grad = True

        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier_conv1', nn.Conv2d(4 + 64, 128, 3, stride=1, padding=2, dilation=2, bias=True))
        self.classifier.add_module('classifier_leaky1', nn.LeakyReLU(0.1, inplace=False))
        self.classifier.add_module('classifier_conv2', nn.Conv2d(128, 32, 1, stride=1, padding=0, bias=True))
        self.classifier.add_module('classifier_leaky2', nn.LeakyReLU(0.1, inplace=False))

        self.classifier.add_module('classifier_conv3', nn.Conv2d(32, 32, 3, stride=1, padding=1, dilation=1, bias=True))
        self.classifier.add_module('classifier_leaky3', nn.LeakyReLU(0.1, inplace=False))
        self.classifier.add_module('classifier_conv4', nn.Conv2d(32, 1, 1, stride=1, padding=0, bias=True))


        self.classifier.requires_grad = True

        if pretrained_weights is not None:
            self.load_state_dict(torch.load(pretrained_weights))

    def forward(self, imgs):
        """
        Run the neural network. imgs should be a list of PIL images, all with the same height/width
        """
        x = self.prep_images(imgs)
        output_size = (x.data.shape[2] / 4, x.data.shape[3] / 4)

        for i, model in enumerate(self.models):
            x = model(x)
            if self.model_name_index[i] == "max_pool_1":
                downsampled_2x = x
            if self.model_name_index[i] == "max_pool_2":
                downsampled_4x = x
                break
            elif self.model_name_index[i] == "max_pool_3":
                downsampled_8x = x
                break
            elif self.model_name_index[i] == "conv8":
                downsampled_8x_2 = x
                break
        # x is the output from the pretrained network at this point

        condensed_2x = self.downsample_2x_condense(downsampled_2x)
        condensed_4x = self.downsample_4x_condense(downsampled_4x)
        # condensed_8x = self.downsample_8x_condense(downsampled_8x)

        if False:
            print()
            print("condensed_2x:", condensed_2x.size())
            print("condensed_4x:", condensed_4x.size())
            print()

        classification_size = downsampled_4x.data.shape[2:4]

        # resized_2x = nn.functional.interpolate(condensed_2x, size=classification_size)
        resized_4x = nn.functional.interpolate(condensed_4x, size=classification_size)
        # resized_8x = nn.functional.interpolate(condensed_8x, size=classification_size)
        resized_2x = condensed_2x
        #resized_4x = condensed_4x
        skip_link_concatenated = torch.cat((resized_2x, resized_4x, ), dim=1)  # stack along the pixel value dimension

        classified = self.classifier(skip_link_concatenated)

        output_size = nn.functional.interpolate(classified, size=output_size)  # resize the output back to the input's origional size
        return output_size



class Yolo2TransferDilated(Yolo2Base):
    """
    This is a playground for dilated convolutions
    """
    def __init__(self, pretrained_weights=None):
        super(Yolo2TransferDilated, self).__init__(pretrained_weights=pretrained_weights)

        self.downsample_2x_condense = nn.Sequential()
        self.downsample_2x_condense.add_module('condense_2x_conv1', nn.Conv2d(32, 8, 2, stride=2, padding=0, bias=True))
        self.downsample_2x_condense.add_module('condense_2x_leaky1', nn.LeakyReLU(0.1, inplace=False))
        self.downsample_2x_condense.add_module('condense_2x_conv2', nn.Conv2d(8, 4, 1, stride=1, padding=0, bias=True))
        self.downsample_2x_condense.add_module('condense_2x_leaky2', nn.LeakyReLU(0.1, inplace=False))
        self.downsample_2x_condense.requires_grad = True


        self.stage_1_1x = nn.Sequential()
        self.stage_1_1x.add_module('condense_4x_conv1', nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=True))
        self.stage_1_1x.add_module('condense_4x_leaky1', nn.LeakyReLU(0.1, inplace=False))
        self.stage_1_1x.add_module('condense_4x_conv2', nn.Conv2d(128, 128, 1, stride=1, padding=0, bias=True))
        self.stage_1_1x.add_module('condense_4x_leaky2', nn.LeakyReLU(0.1, inplace=False))
        self.stage_1_1x.requires_grad = True
        
        self.stage_1_2x = nn.Sequential()
        self.stage_1_2x.add_module('condense_4x_conv1', nn.Conv2d(64, 128, 3, stride=1, padding=2, dilation=2, bias=True))
        self.stage_1_2x.add_module('condense_4x_leaky1', nn.LeakyReLU(0.1, inplace=False))
        self.stage_1_2x.add_module('condense_4x_conv2', nn.Conv2d(128, 128, 1, stride=1, padding=0, bias=True))
        self.stage_1_2x.add_module('condense_4x_leaky2', nn.LeakyReLU(0.1, inplace=False))
        self.stage_1_2x.requires_grad = True

        self.stage_1_combine = nn.Sequential()
        self.stage_1_combine.add_module('stage_1_combine_conv2', nn.Conv2d(128 * 2, 128, 1, stride=1, padding=0, bias=True))
        self.stage_1_combine.add_module('stage_1_combine_leaky2', nn.LeakyReLU(0.1, inplace=False))
        self.stage_1_combine.requires_grad = True



        self.stage_2_1x = nn.Sequential()
        self.stage_2_1x.add_module('maxpool_4x_1', nn.MaxPool2d(2, 2))
        self.stage_2_1x.add_module('condense_4x_conv10', nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=True))
        self.stage_2_1x.add_module('condense_4x_leaky10', nn.LeakyReLU(0.1, inplace=False))
        self.stage_2_1x.add_module('condense_4x_conv11', nn.Conv2d(128, 64, 1, stride=1, padding=0, bias=True))
        self.stage_2_1x.add_module('condense_4x_leaky11', nn.LeakyReLU(0.1, inplace=False))
        self.stage_2_1x.requires_grad = True

        self.stage_2_2x = nn.Sequential()
        self.stage_2_2x.add_module('maxpool_4x_1', nn.MaxPool2d(2, 2))
        self.stage_2_2x.add_module('condense_4x_conv10', nn.Conv2d(128, 128, 3, stride=1, padding=2, dilation=2, bias=True))
        self.stage_2_2x.add_module('condense_4x_leaky10', nn.LeakyReLU(0.1, inplace=False))
        self.stage_2_2x.add_module('condense_4x_conv11', nn.Conv2d(128, 64, 1, stride=1, padding=0, bias=True))
        self.stage_2_2x.add_module('condense_4x_leaky11', nn.LeakyReLU(0.1, inplace=False))
        self.stage_2_2x.requires_grad = True

        self.stage_2_combine = nn.Sequential()
        self.stage_2_combine.add_module('stage_2_combine_conv2', nn.Conv2d(64 * 2, 128, 1, stride=1, padding=0, bias=True))
        self.stage_2_combine.add_module('stage_2_combine_leaky2', nn.LeakyReLU(0.1, inplace=False))
        self.stage_2_combine.requires_grad = True



        self.stage_3_1x = nn.Sequential()
        self.stage_3_1x.add_module('condense_4x_conv12', nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=True))
        self.stage_3_1x.add_module('condense_4x_leaky12', nn.LeakyReLU(0.1, inplace=False))
        self.stage_3_1x.add_module('condense_4x_conv13', nn.Conv2d(128, 64, 1, stride=1, padding=0, bias=True))
        self.stage_3_1x.add_module('condense_4x_leaky13', nn.LeakyReLU(0.1, inplace=False))
        self.stage_3_1x.requires_grad = True

        self.stage_3_2x = nn.Sequential()
        self.stage_3_2x.add_module('condense_4x_conv12', nn.Conv2d(128, 128, 3, stride=1, padding=2, dilation=2, bias=True))
        self.stage_3_2x.add_module('condense_4x_leaky12', nn.LeakyReLU(0.1, inplace=False))
        self.stage_3_2x.add_module('condense_4x_conv13', nn.Conv2d(128, 64, 1, stride=1, padding=0, bias=True))
        self.stage_3_2x.add_module('condense_4x_leaky13', nn.LeakyReLU(0.1, inplace=False))
        self.stage_3_2x.requires_grad = True

        self.stage_3_combine = nn.Sequential()
        self.stage_3_combine.add_module('stage_3_combine_conv2', nn.Conv2d(64 * 2, 64, 1, stride=1, padding=0, bias=True))
        self.stage_3_combine.add_module('stage_3_combine_leaky2', nn.LeakyReLU(0.1, inplace=False))
        self.stage_3_combine.requires_grad = True




        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier_conv1', nn.Conv2d(4 + 64, 128, 3, stride=1, padding=1, bias=True))
        self.classifier.add_module('classifier_leaky1', nn.LeakyReLU(0.1, inplace=False))
        self.classifier.add_module('classifier_conv2', nn.Conv2d(128, 64, 1, stride=1, padding=0, bias=True))
        self.classifier.add_module('classifier_leaky2', nn.LeakyReLU(0.1, inplace=False))
        self.classifier.add_module('classifier_conv3', nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=True))
        self.classifier.add_module('classifier_leaky3', nn.LeakyReLU(0.1, inplace=False))
        self.classifier.add_module('classifier_conv4', nn.Conv2d(128, 1, 1, stride=1, padding=0, bias=True))


        self.classifier.requires_grad = True

        if pretrained_weights is not None:
            self.load_state_dict(torch.load(pretrained_weights))

    def forward(self, imgs):
        """
        Run the neural network. imgs should be a list of PIL images, all with the same height/width
        """
        x = self.prep_images(imgs)
        output_size = (x.data.shape[2] / 4, x.data.shape[3] / 4)

        for i, model in enumerate(self.models):
            x = model(x)
            if self.model_name_index[i] == "max_pool_1":
                downsampled_2x = x
            elif self.model_name_index[i] == "max_pool_2":
                downsampled_4x = x
                break
        # x is the output from the pretrained network at this point

        condensed_2x = self.downsample_2x_condense(downsampled_2x)

        stage_1_1x = self.stage_1_1x(downsampled_4x)
        stage_1_2x = self.stage_1_2x(downsampled_4x)
        stage_1 = self.stage_1_combine(torch.cat((stage_1_1x, stage_1_2x, ), dim=1))
        
        stage_2_1x = self.stage_2_1x(stage_1)
        stage_2_2x = self.stage_2_2x(stage_1)
        stage_2 = self.stage_2_combine(torch.cat((stage_2_1x, stage_2_2x, ), dim=1))

        stage_3_1x = self.stage_3_1x(stage_2)
        stage_3_2x = self.stage_3_2x(stage_2)
        stage_3 = self.stage_3_combine(torch.cat((stage_3_1x, stage_3_2x, ), dim=1))

        classification_size = downsampled_4x.data.shape[2:4]
        resized_stage_3 = nn.functional.interpolate(stage_3, size=classification_size)

        skip_link_concatenated = torch.cat((condensed_2x, resized_stage_3, ), dim=1)  # stack along the pixel value dimension

        classified = self.classifier(skip_link_concatenated)

        output_size = nn.functional.interpolate(classified, size=output_size)  # resize the output back to the input's origional size
        return output_size



class Yolo2TransferUndilated(Yolo2Base):
    """
    This is a playground for dilated convolutions
    """
    def __init__(self, pretrained_weights=None):
        super(Yolo2TransferUndilated, self).__init__(pretrained_weights=pretrained_weights)

        self.downsample_2x_condense = nn.Sequential()
        self.downsample_2x_condense.add_module('condense_2x_conv1', nn.Conv2d(32, 8, 2, stride=2, padding=0, bias=True))
        self.downsample_2x_condense.add_module('condense_2x_leaky1', nn.LeakyReLU(0.1, inplace=False))
        self.downsample_2x_condense.add_module('condense_2x_conv2', nn.Conv2d(8, 4, 1, stride=1, padding=0, bias=True))
        self.downsample_2x_condense.add_module('condense_2x_leaky2', nn.LeakyReLU(0.1, inplace=False))
        self.downsample_2x_condense.requires_grad = True


        self.stage_1_1x = nn.Sequential()
        self.stage_1_1x.add_module('condense_4x_conv1', nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=True))
        self.stage_1_1x.add_module('condense_4x_leaky1', nn.LeakyReLU(0.1, inplace=False))
        self.stage_1_1x.add_module('condense_4x_conv2', nn.Conv2d(128, 128, 1, stride=1, padding=0, bias=True))
        self.stage_1_1x.add_module('condense_4x_leaky2', nn.LeakyReLU(0.1, inplace=False))
        self.stage_1_1x.requires_grad = True
        
        """
        self.stage_1_2x = nn.Sequential()
        self.stage_1_2x.add_module('condense_4x_conv1', nn.Conv2d(64, 128, 3, stride=1, padding=2, dilation=2, bias=True))
        self.stage_1_2x.add_module('condense_4x_leaky1', nn.LeakyReLU(0.1, inplace=False))
        self.stage_1_2x.add_module('condense_4x_conv2', nn.Conv2d(128, 128, 1, stride=1, padding=0, bias=True))
        self.stage_1_2x.add_module('condense_4x_leaky2', nn.LeakyReLU(0.1, inplace=False))
        self.stage_1_2x.requires_grad = True
        """

        self.stage_1_combine = nn.Sequential()
        self.stage_1_combine.add_module('stage_1_combine_conv2', nn.Conv2d(128 * 1, 128, 1, stride=1, padding=0, bias=True))
        self.stage_1_combine.add_module('stage_1_combine_leaky2', nn.LeakyReLU(0.1, inplace=False))
        self.stage_1_combine.requires_grad = True



        self.stage_2_1x = nn.Sequential()
        self.stage_2_1x.add_module('maxpool_4x_1', nn.MaxPool2d(2, 2))
        self.stage_2_1x.add_module('condense_4x_conv10', nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=True))
        self.stage_2_1x.add_module('condense_4x_leaky10', nn.LeakyReLU(0.1, inplace=False))
        self.stage_2_1x.add_module('condense_4x_conv11', nn.Conv2d(128, 64, 1, stride=1, padding=0, bias=True))
        self.stage_2_1x.add_module('condense_4x_leaky11', nn.LeakyReLU(0.1, inplace=False))
        self.stage_2_1x.requires_grad = True

        """
        self.stage_2_2x = nn.Sequential()
        self.stage_2_2x.add_module('maxpool_4x_1', nn.MaxPool2d(2, 2))
        self.stage_2_2x.add_module('condense_4x_conv10', nn.Conv2d(128, 128, 3, stride=1, padding=2, dilation=2, bias=True))
        self.stage_2_2x.add_module('condense_4x_leaky10', nn.LeakyReLU(0.1, inplace=False))
        self.stage_2_2x.add_module('condense_4x_conv11', nn.Conv2d(128, 64, 1, stride=1, padding=0, bias=True))
        self.stage_2_2x.add_module('condense_4x_leaky11', nn.LeakyReLU(0.1, inplace=False))
        self.stage_2_2x.requires_grad = True
        """

        self.stage_2_combine = nn.Sequential()
        self.stage_2_combine.add_module('stage_2_combine_conv2', nn.Conv2d(64 * 1, 128, 1, stride=1, padding=0, bias=True))
        self.stage_2_combine.add_module('stage_2_combine_leaky2', nn.LeakyReLU(0.1, inplace=False))
        self.stage_2_combine.requires_grad = True



        self.stage_3_1x = nn.Sequential()
        self.stage_3_1x.add_module('condense_4x_conv12', nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=True))
        self.stage_3_1x.add_module('condense_4x_leaky12', nn.LeakyReLU(0.1, inplace=False))
        self.stage_3_1x.add_module('condense_4x_conv13', nn.Conv2d(128, 64, 1, stride=1, padding=0, bias=True))
        self.stage_3_1x.add_module('condense_4x_leaky13', nn.LeakyReLU(0.1, inplace=False))
        self.stage_3_1x.requires_grad = True

        """
        self.stage_3_2x = nn.Sequential()
        self.stage_3_2x.add_module('condense_4x_conv12', nn.Conv2d(128, 128, 3, stride=1, padding=2, dilation=2, bias=True))
        self.stage_3_2x.add_module('condense_4x_leaky12', nn.LeakyReLU(0.1, inplace=False))
        self.stage_3_2x.add_module('condense_4x_conv13', nn.Conv2d(128, 64, 1, stride=1, padding=0, bias=True))
        self.stage_3_2x.add_module('condense_4x_leaky13', nn.LeakyReLU(0.1, inplace=False))
        self.stage_3_2x.requires_grad = True
        """

        self.stage_3_combine = nn.Sequential()
        self.stage_3_combine.add_module('stage_3_combine_conv2', nn.Conv2d(64 * 1, 64, 1, stride=1, padding=0, bias=True))
        self.stage_3_combine.add_module('stage_3_combine_leaky2', nn.LeakyReLU(0.1, inplace=False))
        self.stage_3_combine.requires_grad = True




        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier_conv1', nn.Conv2d(4 + 64, 128, 3, stride=1, padding=1, bias=True))
        self.classifier.add_module('classifier_leaky1', nn.LeakyReLU(0.1, inplace=False))
        self.classifier.add_module('classifier_conv2', nn.Conv2d(128, 64, 1, stride=1, padding=0, bias=True))
        self.classifier.add_module('classifier_leaky2', nn.LeakyReLU(0.1, inplace=False))
        self.classifier.add_module('classifier_conv3', nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=True))
        self.classifier.add_module('classifier_leaky3', nn.LeakyReLU(0.1, inplace=False))
        self.classifier.add_module('classifier_conv4', nn.Conv2d(128, 1, 1, stride=1, padding=0, bias=True))


        self.classifier.requires_grad = True

        if pretrained_weights is not None:
            self.load_state_dict(torch.load(pretrained_weights))

    def forward(self, imgs):
        """
        Run the neural network. imgs should be a list of PIL images, all with the same height/width
        """
        x = self.prep_images(imgs)
        output_size = (x.data.shape[2] / 4, x.data.shape[3] / 4)

        for i, model in enumerate(self.models):
            x = model(x)
            if self.model_name_index[i] == "max_pool_1":
                downsampled_2x = x
            elif self.model_name_index[i] == "max_pool_2":
                downsampled_4x = x
                break
        # x is the output from the pretrained network at this point

        condensed_2x = self.downsample_2x_condense(downsampled_2x)

        stage_1_1x = self.stage_1_1x(downsampled_4x)
        # stage_1_2x = self.stage_1_2x(downsampled_4x)
        stage_1 = self.stage_1_combine(torch.cat((stage_1_1x, ), dim=1))
        
        stage_2_1x = self.stage_2_1x(stage_1)
        # stage_2_2x = self.stage_2_2x(stage_1)
        stage_2 = self.stage_2_combine(torch.cat((stage_2_1x, ), dim=1))

        stage_3_1x = self.stage_3_1x(stage_2)
        # stage_3_2x = self.stage_3_2x(stage_2)
        stage_3 = self.stage_3_combine(torch.cat((stage_3_1x, ), dim=1))

        classification_size = downsampled_4x.data.shape[2:4]
        resized_stage_3 = nn.functional.interpolate(stage_3, size=classification_size)

        skip_link_concatenated = torch.cat((condensed_2x, resized_stage_3, ), dim=1)  # stack along the pixel value dimension

        classified = self.classifier(skip_link_concatenated)

        output_size = nn.functional.interpolate(classified, size=output_size)  # resize the output back to the input's origional size
        return output_size



class Yolo2TransferPixelnet(Yolo2Base):
    """
    The goal of this network is to be smaller, using fewer layers than the normal yolo2 transfer network
    """
    def __init__(self, pretrained_weights=None):
        super(Yolo2TransferPixelnet, self).__init__(pretrained_weights=pretrained_weights)

        def gen_nin_module(name, in_num, mid_num, out_num, dilation=1):
            module = nn.Sequential()
            # padding=dilation because convolutions with dilation 1 require padding of 1 pixel,
            # convolutions with dilation 2 require padding of 2 pixels, ect (specifically since the convolution is 3x3)
            module.add_module(name + '_conv1', nn.Conv2d(in_num, mid_num, 3, stride=1, padding=dilation, dilation=dilation, bias=True))
            module.add_module(name + '_leaky1', nn.LeakyReLU(0.1, inplace=False))
            module.add_module(name + '_conv2', nn.Conv2d(mid_num, out_num, 1, stride=1, padding=0, bias=True))
            module.add_module(name + '_leaky2', nn.LeakyReLU(0.1, inplace=False))
            module.requires_grad = True
            module.in_feature_len = in_num
            module.out_feature_len = out_num
            return module

        # input is from maxpool_4 which outputs 64 features

        self.module_1 = gen_nin_module("one", 64, 96, 64)
        self.module_1.add_module('maxpool_4x_1', nn.MaxPool2d(2, 2))

        self.module_2 = gen_nin_module("two", 64, 128, 96)
        self.module_2.add_module('maxpool_4x_1', nn.MaxPool2d(2, 2))

        self.module_3 = gen_nin_module("three", 96, 128, 96)
        self.module_3.add_module('maxpool_4x_1', nn.MaxPool2d(2, 2))

        self.module_4 = gen_nin_module("four", 96, 128, 128)
        self.module_4.add_module('maxpool_4x_1', nn.MaxPool2d(2, 2))

        self.module_5 = gen_nin_module("five", 128, 196, 32)


        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier_pixelwise_conv1', nn.Conv2d(64 + self.module_1.out_feature_len +
                                                                       self.module_3.out_feature_len + self.module_5.out_feature_len,
                                                                       64, 1, stride=1, padding=0, bias=True))
        self.classifier.add_module('classifier_pixelwise_leaky1', nn.LeakyReLU(0.1, inplace=False))
        self.classifier.add_module('classifier_pixelwise_conv2', nn.Conv2d(64, 1, 1, stride=1, padding=0, bias=True))
        self.classifier.requires_grad = True


        if pretrained_weights is not None:
            self.load_state_dict(torch.load(pretrained_weights))

    def forward(self, imgs):
        """
        Run the neural network. imgs should be a list of PIL images, all with the same height/width
        """
        x = self.prep_images(imgs)
        output_size = (x.data.shape[2] / 4, x.data.shape[3] / 4)

        for i, layer in enumerate(self.models):
            x = layer(x)
            if self.model_name_index[i] == "max_pool_2":
                downsampled_4x = x
                break

        output_1 = self.module_1(downsampled_4x)
        output_2 = self.module_2(output_1)
        output_3 = self.module_3(output_2)
        output_4 = self.module_4(output_3)
        output_5 = self.module_5(output_4)

        if False:
            print()
            print("downsampled_4x:", downsampled_4x.size())
            print("output_1:", output_1.size())
            print("output_2:", output_2.size())
            print("output_3:", output_3.size())
            print("output_4:", output_4.size())
            print("output_5:", output_5.size())
            print("output_size:", output_size)
            print()

        classification_size = downsampled_4x.data.shape[2:4]

        # resized_yolo_out = nn.functional.interpolate(downsampled_4x, size=classification_size)
        resized_yolo_out = downsampled_4x
        resized_1 = nn.functional.interpolate(output_1, size=classification_size, mode='bilinear', align_corners=False)
        # resized_2 = nn.functional.interpolate(output_2, size=classification_size, mode='bilinear', align_corners=False)
        resized_3 = nn.functional.interpolate(output_3, size=classification_size, mode='bilinear', align_corners=False)
        # resized_4 = nn.functional.interpolate(output_4, size=classification_size, mode='bilinear', align_corners=False)
        resized_5 = nn.functional.interpolate(output_5, size=classification_size, mode='bilinear', align_corners=False)

        skip_link_concatenated = torch.cat((resized_yolo_out, resized_1, resized_3, resized_5), dim=1)  # stack along the pixel value dimension

        classified = self.classifier(skip_link_concatenated)

        # output_size = nn.functional.interpolate(classified, size=output_size)  # resize the output back to the input's origional size
        output_size = classified
        return output_size



class Yolo2TransferPixelnetMoreYolo(Yolo2Base):
    """
    The goal of this network is to be smaller, using fewer layers than the normal yolo2 transfer network
    """
    def __init__(self, pretrained_weights=None, init_yolo_weights=True):
        super(Yolo2TransferPixelnetMoreYolo, self).__init__(pretrained_weights=pretrained_weights, init_weights=init_yolo_weights)

        def gen_nin_module(name, in_num, mid_num, out_num, dilation=1):
            module = nn.Sequential()
            # padding=dilation because convolutions with dilation 1 require padding of 1 pixel,
            # convolutions with dilation 2 require padding of 2 pixels, ect (specifically since the convolution is 3x3)
            module.add_module(name + '_conv1', nn.Conv2d(in_num, mid_num, 3, stride=1, padding=dilation, dilation=dilation, bias=True))
            module.add_module(name + '_leaky1', nn.LeakyReLU(0.1, inplace=False))
            module.add_module(name + '_conv2', nn.Conv2d(mid_num, out_num, 1, stride=1, padding=0, bias=True))
            module.add_module(name + '_leaky2', nn.LeakyReLU(0.1, inplace=False))
            module.requires_grad = True
            module.in_feature_len = in_num
            module.out_len = out_num
            return module

        def gen_linear_compress_module(name, in_module, out_num):
            try:
                in_len = in_module.out_len
            except:
                in_len = in_module
            module = nn.Sequential()
            module.add_module('compress_' + name + '_conv1', nn.Conv2d(in_len, out_num, 1, stride=1, padding=0, bias=True))
            module.requires_grad = True
            module.in_feature_len = in_len
            module.out_len = out_num
            return module

        # input is from maxpool_4 which outputs 64 features

        self.compress_4x = gen_linear_compress_module("yolo_4x", 64, 16)
        self.compress_8x = gen_linear_compress_module("yolo_8x", 128, 16)

        self.module_1 = gen_nin_module("one", 128, 196, 96)
        self.module_1.add_module('maxpool_4x_1', nn.MaxPool2d(2, 2))
        self.compress_1 = gen_linear_compress_module("one", self.module_1, 16)

        self.module_2 = gen_nin_module("two", 96, 128, 96)
        self.module_2.add_module('maxpool_4x_1', nn.MaxPool2d(2, 2))
        self.compress_2 = gen_linear_compress_module("two", self.module_2, 16)

        self.module_3 = gen_nin_module("three", 96, 196, 128)
        self.module_3.add_module('maxpool_4x_1', nn.MaxPool2d(2, 2))
        self.compress_3 = gen_linear_compress_module("three", self.module_3, 16)

        self.module_4 = gen_nin_module("four", 128, 196, 32)


        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier_pixelwise_conv1', nn.Conv2d(self.compress_4x.out_len +
                                                                       self.compress_8x.out_len +
                                                                       self.compress_1.out_len +
                                                                       self.compress_2.out_len +
                                                                       self.compress_3.out_len +
                                                                       self.module_4.out_len,
                                                                       64, 1, stride=1, padding=0, bias=True))
        self.classifier.add_module('classifier_pixelwise_leaky1', nn.LeakyReLU(0.1, inplace=False))
        self.classifier.add_module('classifier_pixelwise_conv2', nn.Conv2d(64, 1, 1, stride=1, padding=0, bias=True))
        self.classifier.requires_grad = True

        if pretrained_weights is not None:
            self.load_state_dict(torch.load(pretrained_weights))

    def forward(self, imgs):
        """
        Run the neural network. imgs should be a list of PIL images, all with the same height/width
        """
        x = self.prep_images(imgs)
        output_size = (x.data.shape[2] / 4, x.data.shape[3] / 4)

        for i, layer in enumerate(self.models):
            x = layer(x)
            if self.model_name_index[i] == "max_pool_2":
                downsampled_4x = x
            if self.model_name_index[i] == "max_pool_3":
                downsampled_8x = x
                break

        output_1 = self.module_1(downsampled_8x)
        output_2 = self.module_2(output_1)
        output_3 = self.module_3(output_2)
        output_4 = self.module_4(output_3)
        # output_5 = self.module_5(output_4)

        if False:
            print()
            print("downsampled_4x:", downsampled_4x.size())
            print("output_1:", output_1.size())
            print("output_2:", output_2.size())
            print("output_3:", output_3.size())
            print("output_4:", output_4.size())
            print("output_size:", output_size)
            print()

        classification_size = downsampled_4x.data.shape[2:4]

        # resized_yolo_out = nn.functional.interpolate(downsampled_4x, size=classification_size)
        resized_yolo_4x = downsampled_4x
        resized_yolo_8x = nn.functional.interpolate(downsampled_8x, size=classification_size, mode='bilinear', align_corners=False)
        resized_1 = nn.functional.interpolate(output_1, size=classification_size, mode='bilinear', align_corners=False)
        resized_2 = nn.functional.interpolate(output_2, size=classification_size, mode='bilinear', align_corners=False)
        resized_3 = nn.functional.interpolate(output_3, size=classification_size, mode='bilinear', align_corners=False)
        resized_4 = nn.functional.interpolate(output_4, size=classification_size, mode='bilinear', align_corners=False)

        skip_link_concatenated = torch.cat((self.compress_4x(resized_yolo_4x),
                                            self.compress_8x(resized_yolo_8x),
                                            self.compress_1(resized_1),
                                            self.compress_2(resized_2),
                                            self.compress_3(resized_3),
                                            resized_4), dim=1)  # stack along the pixel value dimension

        classified = self.classifier(skip_link_concatenated)

        # output_size = nn.functional.interpolate(classified, size=output_size)  # resize the output back to the input's origional size
        output_size = classified
        return output_size



class PixelnetNoYolo(Yolo2Base):
    """
    The goal of this network is to be smaller, using fewer layers than the normal yolo2 transfer network
    """
    def __init__(self, pretrained_weights=None, init_yolo_weights=True):
        # super(PixelnetNoYolo, self).__init__()
        super(PixelnetNoYolo, self).__init__(pretrained_weights=pretrained_weights, init_weights=init_yolo_weights)

        def gen_nin_module(name, in_num, mid_num, out_num, dilation=1):
            module = nn.Sequential()
            # padding=dilation because convolutions with dilation 1 require padding of 1 pixel,
            # convolutions with dilation 2 require padding of 2 pixels, ect (specifically since the convolution is 3x3)
            module.add_module(name + '_conv1', nn.Conv2d(in_num, mid_num, 3, stride=1, padding=dilation, dilation=dilation, bias=True))
            module.add_module(name + '_leaky1', nn.LeakyReLU(0.1, inplace=False))
            module.add_module(name + '_conv2', nn.Conv2d(mid_num, out_num, 1, stride=1, padding=0, bias=True))
            module.add_module(name + '_leaky2', nn.LeakyReLU(0.1, inplace=False))
            module.requires_grad = True
            module.in_feature_len = in_num
            module.out_len = out_num
            return module

        def gen_linear_compress_module(name, in_module, out_num):
            try:
                in_len = in_module.out_len
            except:
                in_len = in_module
            module = nn.Sequential()
            module.add_module('compress_' + name + '_conv1', nn.Conv2d(in_len, out_num, 1, stride=1, padding=0, bias=True))
            module.requires_grad = True
            module.in_feature_len = in_len
            module.out_len = out_num
            return module

        # input is from maxpool_4 which outputs 64 features
        self.compress_4x = gen_linear_compress_module("yolo_4x", 64, 16)
        self.compress_8x = gen_linear_compress_module("yolo_8x", 128, 16)

        self.models = torch.nn.ModuleList()
        self.model_name_index = {}

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

        self.module_1 = gen_nin_module("one", 128, 196, 96)
        self.module_1.add_module('maxpool_4x_1', nn.MaxPool2d(2, 2))
        self.compress_1 = gen_linear_compress_module("one", self.module_1, 16)

        self.module_2 = gen_nin_module("two", 96, 128, 96)
        self.module_2.add_module('maxpool_4x_1', nn.MaxPool2d(2, 2))
        self.compress_2 = gen_linear_compress_module("two", self.module_2, 16)

        self.module_3 = gen_nin_module("three", 96, 196, 128)
        self.module_3.add_module('maxpool_4x_1', nn.MaxPool2d(2, 2))
        self.compress_3 = gen_linear_compress_module("three", self.module_3, 16)

        self.module_4 = gen_nin_module("four", 128, 196, 32)


        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier_pixelwise_conv1', nn.Conv2d(self.compress_4x.out_len +
                                                                       self.compress_8x.out_len +
                                                                       self.compress_1.out_len +
                                                                       self.compress_2.out_len +
                                                                       self.compress_3.out_len +
                                                                       self.module_4.out_len,
                                                                       64, 1, stride=1, padding=0, bias=True))
        self.classifier.add_module('classifier_pixelwise_leaky1', nn.LeakyReLU(0.1, inplace=False))
        self.classifier.add_module('classifier_pixelwise_conv2', nn.Conv2d(64, 1, 1, stride=1, padding=0, bias=True))
        self.classifier.requires_grad = True

        if pretrained_weights is not None:
            self.load_state_dict(torch.load(pretrained_weights))

    def do_images(self, imgs):
        """
        Run the neural network. imgs should be a list of PIL images, all with the same height/width
        """
        x = self.prep_images(imgs)
        return self.forward(x)

    def forward(self, x):
        # x = x.type(torch.cuda.HalfTensor)
        output_size = (x.data.shape[2] / 4, x.data.shape[3] / 4)

        for i, layer in enumerate(self.models):
            x = layer(x)
            if self.model_name_index[i] == "max_pool_2":
                downsampled_4x = x
            if self.model_name_index[i] == "max_pool_3":
                downsampled_8x = x
                break

        output_1 = self.module_1(downsampled_8x)
        output_2 = self.module_2(output_1)
        output_3 = self.module_3(output_2)
        output_4 = self.module_4(output_3)
        # output_5 = self.module_5(output_4)

        if False:
            print()
            print("downsampled_4x:", downsampled_4x.size())
            print("output_1:", output_1.size())
            print("output_2:", output_2.size())
            print("output_3:", output_3.size())
            print("output_4:", output_4.size())
            print("output_size:", output_size)
            print()

        classification_size = downsampled_4x.data.shape[2:4]

        # resized_yolo_out = nn.functional.interpolate(downsampled_4x, size=classification_size)
        resized_yolo_4x = downsampled_4x
        resized_yolo_8x = nn.functional.interpolate(downsampled_8x, size=classification_size, mode='bilinear', align_corners=False)
        resized_1 = nn.functional.interpolate(output_1, size=classification_size, mode='bilinear', align_corners=False)
        resized_2 = nn.functional.interpolate(output_2, size=classification_size, mode='bilinear', align_corners=False)
        resized_3 = nn.functional.interpolate(output_3, size=classification_size, mode='bilinear', align_corners=False)
        resized_4 = nn.functional.interpolate(output_4, size=classification_size, mode='bilinear', align_corners=False)

        skip_link_concatenated = torch.cat((self.compress_4x(resized_yolo_4x),
                                            self.compress_8x(resized_yolo_8x),
                                            self.compress_1(resized_1),
                                            self.compress_2(resized_2),
                                            self.compress_3(resized_3),
                                            resized_4), dim=1)  # stack along the pixel value dimension

        classified = self.classifier(skip_link_concatenated)

        # output_size = nn.functional.interpolate(classified, size=output_size)  # resize the output back to the input's origional size
        output_size = classified
        return output_size






class PixelnetNoYoloBottleneck(torch.nn.Module):
    """

    The goal of this network is to be smaller, using fewer layers than the normal yolo2 transfer network
    """
    def __init__(self):
        super(PixelnetNoYoloBottleneck, self).__init__()

        def gen_nin_module(name, in_num, mid_num, out_num, dilation=1):
            module = nn.Sequential()
            # padding=dilation because convolutions with dilation 1 require padding of 1 pixel,
            # convolutions with dilation 2 require padding of 2 pixels, ect (specifically since the convolution is 3x3)
            module.add_module(name + '_conv1', nn.Conv2d(in_num, mid_num, 3, stride=1, padding=dilation, dilation=dilation, bias=True))
            module.add_module(name + '_leaky1', nn.LeakyReLU(0.1, inplace=False))
            module.add_module(name + '_conv2', nn.Conv2d(mid_num, out_num, 1, stride=1, padding=0, bias=True))
            module.add_module(name + '_leaky2', nn.LeakyReLU(0.1, inplace=False))
            module.requires_grad = True
            module.in_feature_len = in_num
            module.out_len = out_num
            return module

        def gen_linear_compress_module(name, in_module, out_num):
            try:
                in_len = in_module.out_len
            except:
                in_len = in_module
            module = nn.Sequential()
            module.add_module('compress_' + name + '_conv1', nn.Conv2d(in_len, out_num, 1, stride=1, padding=0, bias=True))
            module.requires_grad = True
            module.in_feature_len = in_len
            module.out_len = out_num
            return module

        def bottleneck_block(x, expand=64, squeeze=16):
            module = nn.Sequential()

            # m = Conv2D(expand, (1,1))(x)
            module.add_module('bottleneck_' + name + '_conv1', nn.Conv2d(in_len, expand, 1, stride=1, padding=0, bias=True))
            # m = BatchNormalization()(m)
            module.add_module('bottleneck_' + name + '_batchnorm1', nn.BatchNorm2d(expand))
            # m = Activation('relu6')(m)
            module.add_module('bottleneck_' + name + '_batchnorm1', nn.relu6())
            # m = DepthwiseConv2D((3,3))(m)
            m = BatchNormalization()(m)
            m = Activation('relu6')(m)
            m = Conv2D(squeeze, (1,1))(m)
            m = BatchNormalization()(m)
            m = Add()([m, x])

            module.requires_grad = True
            return 

        # input is from maxpool_4 which outputs 64 features
        self.compress_4x = gen_linear_compress_module("yolo_4x", 64, 16)
        self.compress_8x = gen_linear_compress_module("yolo_8x", 128, 16)

        self.models = torch.nn.ModuleList()
        self.model_name_index = {}

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

        self.module_1 = gen_nin_module("one", 128, 196, 96)
        self.module_1.add_module('maxpool_4x_1', nn.MaxPool2d(2, 2))
        self.compress_1 = gen_linear_compress_module("one", self.module_1, 16)

        self.module_2 = gen_nin_module("two", 96, 128, 96)
        self.module_2.add_module('maxpool_4x_1', nn.MaxPool2d(2, 2))
        self.compress_2 = gen_linear_compress_module("two", self.module_2, 16)

        self.module_3 = gen_nin_module("three", 96, 196, 128)
        self.module_3.add_module('maxpool_4x_1', nn.MaxPool2d(2, 2))
        self.compress_3 = gen_linear_compress_module("three", self.module_3, 16)

        self.module_4 = gen_nin_module("four", 128, 196, 32)


        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier_pixelwise_conv1', nn.Conv2d(self.compress_4x.out_len +
                                                                       self.compress_8x.out_len +
                                                                       self.compress_1.out_len +
                                                                       self.compress_2.out_len +
                                                                       self.compress_3.out_len +
                                                                       self.module_4.out_len,
                                                                       64, 1, stride=1, padding=0, bias=True))
        self.classifier.add_module('classifier_pixelwise_leaky1', nn.LeakyReLU(0.1, inplace=False))
        self.classifier.add_module('classifier_pixelwise_conv2', nn.Conv2d(64, 1, 1, stride=1, padding=0, bias=True))
        self.classifier.requires_grad = True

        if pretrained_weights is not None:
            self.load_state_dict(torch.load(pretrained_weights))

    def do_images(self, imgs):
        """
        Run the neural network. imgs should be a list of PIL images, all with the same height/width
        """
        x = self.prep_images(imgs)

        return self.forward(x)

    def forward(self, x):
        # x = x.type(torch.cuda.HalfTensor)
        output_size = (x.data.shape[2] / 4, x.data.shape[3] / 4)

        for i, layer in enumerate(self.models):
            x = layer(x)
            if self.model_name_index[i] == "max_pool_2":
                downsampled_4x = x
            if self.model_name_index[i] == "max_pool_3":
                downsampled_8x = x
                break

        output_1 = self.module_1(downsampled_8x)
        output_2 = self.module_2(output_1)
        output_3 = self.module_3(output_2)
        output_4 = self.module_4(output_3)
        # output_5 = self.module_5(output_4)

        if False:
            print()
            print("downsampled_4x:", downsampled_4x.size())
            print("output_1:", output_1.size())
            print("output_2:", output_2.size())
            print("output_3:", output_3.size())
            print("output_4:", output_4.size())
            print("output_size:", output_size)
            print()

        classification_size = downsampled_4x.data.shape[2:4]

        # resized_yolo_out = nn.functional.interpolate(downsampled_4x, size=classification_size)
        resized_yolo_4x = downsampled_4x
        resized_yolo_8x = nn.functional.interpolate(downsampled_8x, size=classification_size, mode='bilinear', align_corners=False)
        resized_1 = nn.functional.interpolate(output_1, size=classification_size, mode='bilinear', align_corners=False)
        resized_2 = nn.functional.interpolate(output_2, size=classification_size, mode='bilinear', align_corners=False)
        resized_3 = nn.functional.interpolate(output_3, size=classification_size, mode='bilinear', align_corners=False)
        resized_4 = nn.functional.interpolate(output_4, size=classification_size, mode='bilinear', align_corners=False)

        skip_link_concatenated = torch.cat((self.compress_4x(resized_yolo_4x),
                                            self.compress_8x(resized_yolo_8x),
                                            self.compress_1(resized_1),
                                            self.compress_2(resized_2),

                                            self.compress_3(resized_3),
                                            resized_4), dim=1)  # stack along the pixel value dimension

        classified = self.classifier(skip_link_concatenated)

        # output_size = nn.functional.interpolate(classified, size=output_size)  # resize the output back to the input's origional size
        output_size = classified
        return output_size



