import numpy as np
import cv2
import random
import sys


def main():
    try:
        input_file_list = sys.argv[1]
    except:
        print("incorrect usage. Input file list must be the only argument")
        return
    # training data
    gen_from_category(input_file_list, "train")

    # testing data
    gen_from_category(input_file_list, "test")


def gen_from_category(input_file_list, filter):
    """
    Reads images, generates descriptors, and saves them in a .npz file.
    Image paths are read from the file input_file_list
    Only paths that have the string filter in them will be read. This is used to
    distinguish between the training and testing set.
    """
    descriptors = {}
    input_files = open(input_file_list, "r")
    paths = list(map(lambda x: x.strip(), input_files))
    random.shuffle(paths)  # generate (and save) the image descriptors in a random order
    for image_path in paths:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is not None and image_path.find(filter) != -1:
            print("working on image " + image_path)
            descriptors[image_path] = calc_descriptor(image)

    # I'm using an npz file instead of a pickle file because I'm more familiar with them.
    # They work just as well for this application.
    np.savez("image_descriptors_" + filter + ".npz", **descriptors)


def calc_descriptor(image, t=6, b_w=4, b_h=4):
    """
    Generate and return a feature vector from the passed image.
    """
    W = image.shape[1]
    H = image.shape[0]
    delta_w = W // (b_w + 1)
    delta_h = H // (b_h + 1)

    histogram_list = []  # store all the partial histogram vectors.
    # They will be concatenated into a single numpy array after descriptors for each image region are extracted

    for y in range(0, b_h):
        lower_y = y * delta_h
        upper_y = (y + 2) * delta_h
        for x in range(0, b_w):
            lower_x = x * delta_w
            upper_x = (x + 2) * delta_w

            subimage = image[lower_y : upper_y, lower_x : upper_x, :]
            H, edges = np.histogramdd(subimage.reshape((-1, 3)), (t, t, t), ((0, 255), (0, 255), (0, 255)))
            histogram_list.append(H.reshape((-1)))  # reshape((-1)) flattens

    return np.concatenate(histogram_list, axis=0)


if __name__ == "__main__":
    main()