import cv2
import numpy as np
import matplotlib.pyplot as plt
from debug_utils import imshow
import os
from itertools import *


hog = cv2.HOGDescriptor()

block_size = 50


def train(train_img, test_img):
    img_blocks = []
    ans = []

    scales = [1, 0.5, 0.25]
    

    for img, ans_img in zip(train_img, test_img):
        blocks, coords = break_image_into_blocks(img, block_size, overlap=0.25)
        
        for coord in coords:
            if ans_img[coord[0], coord[1]].max() == 0:
                ans.append(0)
                img_blocks.append(blocks.pop(0))
                print("black point")
            elif ans_img[coords[0], coord[1]].min() == 255:
                ans.append(1)
                img_blocks.append(blocks.pop(0))
                print("white point*****")
            else:
                pass  # the point is marked as non-usefull
                blocks.pop(0)
                print("grey point********************************")
    

        
    

def process_img(weights, image):
    return 0


def main():
    traindata = [[cv2.imread(os.path.join("igvc_sim_trainset3", im), cv2.IMREAD_COLOR), cv2.imread(os.path.join("igvc_sim_trainset3_gt", im), cv2.IMREAD_COLOR)] for im in os.listdir("igvc_sim_trainset3")]
    trainimg, traingt = zip(*traindata)
    train(trainimg, traingt)
    result = process_img(None, img)


"""
compute(...)
 |      compute(img[, winStride[, padding[, locations]]]) -> descriptors
 |      .   @brief Computes HOG descriptors of given image.
 |      .   @param img Matrix of the type CV_8U containing an image where HOG features will be calculated.
 |      .   @param descriptors Matrix of the type CV_32F
 |      .   @param winStride Window stride. It must be a multiple of block stride.
 |      .   @param padding Padding
 |      .   @param locations Vector of Point

"""

#########################################
# utilites
#########################################


def break_image_into_blocks(img, blocksize, overlap=1.):
    step = int(blocksize * overlap)
    blocks = []
    locations = []
    for i in xrange((img.shape[0] - blocksize) / step + 1):
        for j in xrange((img.shape[1] - blocksize) / step + 1):
            blocks.append(img[(i * step): (i * step) + blocksize, (j * step): (j * step) + blocksize])
            locations.append([(i * step) + blocksize / 2, (j * step) + blocksize / 2])
    return blocks, locations

if __name__ == '__main__':
    main()