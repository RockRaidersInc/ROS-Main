from __future__ import print_function
import time

import torch
from torch import nn
import math
import PIL

from nn_utils import *


def conv_bn_back(inp, oup, stride):
    out = nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        # nn.ReLU6(inplace=True)
        nn.ELU()
    )
    out.in_feature_len = inp
    out.out_len = oup
    return out


def conv_1x1_bn_back(inp, oup):
    out = nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        # nn.ReLU6(inplace=True)
        nn.ELU()
    )
    out.in_feature_len = inp
    out.out_len = oup
    return out


def make_divisible_back(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)



class InvertedResidual_no(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=False),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=False),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=False),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        
        self.in_feature_len = inp
        self.out_len = oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)




def conv_bn(inp, oup, stride):
    out = nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        # nn.ReLU6(inplace=True)
        nn.ELU()
    )
    out.in_feature_len = inp
    out.out_len = oup
    return out


def conv_1x1_bn(inp, oup):
    out = nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        # nn.ReLU6(inplace=True)
        nn.ELU()
    )
    out.in_feature_len = inp
    out.out_len = oup
    return out


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)



def InvertedResidual_no2(inp, oup, stride=1, expand_ratio=6):
    assert stride in [1, 2]
    hidden_dim = int(inp * expand_ratio)
    use_res_connect = stride == 1 and inp == oup

    conv = nn.Sequential()
    # pw
    conv.add_module("a", nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
    conv.add_module("b", nn.BatchNorm2d(hidden_dim))
    conv.add_module("c", nn.ELU())
    # dw
    conv.add_module("d", nn.Conv2d(hidden_dim, hidden_dim, 3, stride, padding={1: 1, 2: 2}[stride], bias=False))
    conv.add_module("e", nn.BatchNorm2d(hidden_dim))
    conv.add_module("f", nn.ELU())
    # pw-linear
    conv.add_module("g", nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
    conv.add_module("h", nn.BatchNorm2d(oup))
    conv.add_module("i", nn.ELU())

    conv.in_feature_len = inp
    conv.out_len = oup
    conv.requires_grad = True

    return conv

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=False),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=False),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=False),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        
        self.in_feature_len = inp
        self.out_len = oup
        self.conv.requires_grad = True

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PixelMobileNet(torch.nn.Module):
    """
    The goal of this network is to be smaller, using fewer layers than the normal yolo2 transfer network
    """
    def __init__(self, pretrained_weights=None):
        super(PixelMobileNet, self).__init__()

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

        def add_to_module(name, ops):
            module = nn.Sequential()
            for i, op in enumerate(ops):
                module.add_module(name + " " + str(i), op)
            module.requires_grad = True
            module.in_feature_len = ops[0].in_feature_len
            module.out_len = ops[-1].out_len
            return module

        self.module_1 = add_to_module("module_1", [conv_bn(3, 24, 2)])
        #  t,  c, n, s
        # [1, 16, 1, 1],
        self.module_2 = InvertedResidual(24, 16, stride=2, expand_ratio=1)
        # [6, 24, 2, 2],
        module_3_1 = InvertedResidual(self.module_2.out_len, 32, stride=2, expand_ratio=6)
        module_3_2 = InvertedResidual(module_3_1.out_len, 32, stride=1, expand_ratio=6)
        self.module_3 = add_to_module("module_3", [module_3_1, module_3_2])
        # [6, 32, 3, 2],
        module_4_1 = InvertedResidual(module_3_2.out_len, 32, stride=2, expand_ratio=6)
        module_4_2 = InvertedResidual(module_4_1.out_len, 32, stride=1, expand_ratio=6)
        # module_4_3 = InvertedResidual(module_4_2.out_len, 32, stride=1, expand_ratio=6)
        self.module_4 = add_to_module("module_4", [module_4_1, module_4_2])
        # [6, 64, 4, 2],
        module_5_1 = InvertedResidual(module_4_2.out_len, 64, stride=2, expand_ratio=6)
        module_5_2 = InvertedResidual(module_5_1.out_len, 64, stride=1, expand_ratio=6)
        module_5_3 = InvertedResidual(module_5_2.out_len, 64, stride=1, expand_ratio=6)
        # module_5_4 = InvertedResidual(module_5_3.out_len, 64, stride=1, expand_ratio=6)
        self.module_5 = add_to_module("module_5", [module_5_1, module_5_2, module_5_3])
        # [6, 96, 3, 1],
        module_6_1 = InvertedResidual(module_5_2.out_len, 96, stride=2, expand_ratio=6)
        module_6_2 = InvertedResidual(module_6_1.out_len, 96, stride=1, expand_ratio=6)
        # module_6_3 = InvertedResidual(module_6_2.out_len, 96, stride=1, expand_ratio=6)
        self.module_6 = add_to_module("module_6", [module_6_1, module_6_2])
        # [6, 160, 3, 2],
        module_7_1 = InvertedResidual(module_6_1.out_len, 160, stride=2, expand_ratio=6)
        module_7_2 = InvertedResidual(module_7_1.out_len, 160, stride=1, expand_ratio=6)
        # module_7_3 = InvertedResidual(module_7_2.out_len, 160, stride=1, expand_ratio=6)
        self.module_7 = add_to_module("module_6", [module_7_1, module_7_2])
        # [6, 320, 1, 1],
        # self.module_8 = InvertedResidual(module_7_2.out_len, 320, stride=1, expand_ratio=6)

        self.compress_1 = gen_linear_compress_module("compress_1", self.module_1.out_len, 1)
        self.compress_2 = gen_linear_compress_module("compress_2", self.module_2.out_len, 8)
        self.compress_3 = gen_linear_compress_module("compress_3", self.module_3.out_len, 8)
        self.compress_4 = gen_linear_compress_module("compress_4", self.module_4.out_len, 8)
        self.compress_5 = gen_linear_compress_module("compress_5", self.module_5.out_len, 8)
        self.compress_6 = gen_linear_compress_module("compress_6", self.module_6.out_len, 8)
        self.compress_7 = gen_linear_compress_module("compress_7", self.module_7.out_len, 8)
        # self.compress_8 = gen_linear_compress_module("compress_8", self.module_8.out_len, 16)



        self.classifier_1 = nn.Sequential()
        self.classifier_1.out_len = 1

        self.classifier_1.add_module('classifier_1_pixelwise_conv1', nn.Conv2d(
                                                                       # self.compress_1.out_len +
                                                                       self.compress_2.out_len +
                                                                       self.compress_3.out_len +
                                                                       self.compress_4.out_len +
                                                                       self.compress_5.out_len +
                                                                       self.compress_6.out_len +
                                                                       self.compress_7.out_len,
                                                                       # self.compress_8.out_len,
                                                                       64, 1, stride=1, padding=0, bias=True))
        self.classifier_1.add_module('classifier_1_pixelwise_leaky1', nn.ELU())
        self.classifier_1.add_module('classifier_1_pixelwise_conv2', nn.Conv2d(64, self.classifier_1.out_len, 1, stride=1, padding=0, bias=True))
        self.classifier_1.requires_grad = True

        if pretrained_weights is not None:
            self.load_state_dict(torch.load(pretrained_weights))


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

            # img = img_buffer.view(img.shape[0], img.shape[1], 3).transpose(0,1).transpose(0,2).contiguous()
            img = img_buffer.view(img.shape[0], img.shape[1], 3).transpose(0,1).transpose(0,2)
            # img = img.view(3, height, width)
            xs.append(img.float())

        x = torch.stack(xs).div(255.0)
        return x

    def do_images(self, imgs):
        """
        Run the neural network. imgs should be a list of PIL images, all with the same height/width
        """
        x = self.prep_images(imgs)
        return self.forward(x)


    def resize_and_classify(self, input_list, classifier, output_size=None):
        if output_size is None:
            max_dim_2 = max(map(lambda x: x.data.shape[2], input_list))
            max_dim_3 = max(map(lambda x: x.data.shape[3], input_list))
            output_size = (max_dim_2, max_dim_3)

        resized = list(map(lambda x: nn.functional.interpolate(x, size=output_size, mode='bilinear', align_corners=True), input_list))

        concatenated = torch.cat(resized, dim=1)
        output = classifier(concatenated)

        return output
        


    def forward(self, x):
        output_size = (int(x.data.shape[2] / 4), int(x.data.shape[3] / 4))
        

        output_1 = self.module_1(x)
        output_2 = self.module_2(output_1)
        output_3 = self.module_3(output_2)
        output_4 = self.module_4(output_3)
        output_5 = self.module_5(output_4)
        output_6 = self.module_6(output_5)
        output_7 = self.module_7(output_6)
        # output_8 = self.module_8(output_7)

        if False:
            print()
            print("output_1:", output_1.size())
            print("output_2:", output_2.size())
            print("output_3:", output_3.size())
            print("output_4:", output_4.size())
            print("output_5:", output_5.size())
            print("output_6:", output_6.size())
            print("output_7:", output_7.size())
            # print("output_8:", output_8.size())
            print()

        # compressed_1 = self.compress_1(output_1)
        compressed_2 = self.compress_2(output_2)
        compressed_3 = self.compress_3(output_3)
        compressed_4 = self.compress_4(output_4)
        compressed_5 = self.compress_5(output_5)
        compressed_6 = self.compress_6(output_6)
        compressed_7 = self.compress_7(output_7)
        # compressed_8 = self.compress_8(output_8)

        # # resized_1 = nn.functional.interpolate(compressed_1, size=output_size, mode='bilinear', align_corners=False)
        # resized_2 = nn.functional.interpolate(compressed_2, size=output_size, mode='bilinear', align_corners=False)
        # resized_3 = nn.functional.interpolate(compressed_3, size=output_size, mode='bilinear', align_corners=False)
        # resized_4 = nn.functional.interpolate(compressed_4, size=output_size, mode='bilinear', align_corners=False)
        # resized_5 = nn.functional.interpolate(compressed_5, size=output_size, mode='bilinear', align_corners=False)
        # resized_6 = nn.functional.interpolate(compressed_6, size=output_size, mode='bilinear', align_corners=False)
        # resized_7 = nn.functional.interpolate(compressed_7, size=output_size, mode='bilinear', align_corners=False)
        # # resized_8 = nn.functional.interpolate(compressed_8, size=output_size, mode='bilinear', align_corners=False)
        # skip_link_concatenated = torch.cat((resized_2, resized_3, resized_4, resized_5, resized_6, resized_7), dim=1)  # stack along the pixel value dimension
        # classified = self.classifier(skip_link_concatenated)

        classified_1 = self.resize_and_classify([compressed_2, compressed_3, compressed_4, compressed_5, compressed_6, compressed_7], self.classifier_1, output_size=output_size)

        # output_size = nn.functional.interpolate(classified, size=output_size)  # resize the output back to the input's origional size
        output_size = classified_1
        return output_size



