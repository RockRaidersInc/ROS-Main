import random
import numpy as np
import cv2
import torch
import os
import functools
import collections

from confusion_mat_tools import save_confusion_matrix


np.random.seed(0)


label_name_to_onehot = {"other" :     np.array([1, 0]),
                        "lane":      np.array([0, 1])}

label_name_to_index = {"other" :     0,
                        "lane":     1}

index_to_label_name = {0: "other",
                       1: "lane"}


def to_indices(arr):
    return np.array(list(map(lambda ar: ar.argmax(), [arr[i] for i in range(arr.shape[0])])))


# from https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
class memoized(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''
    def __init__(self, func):
        self.func = func
        self.cache = {}
    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value
    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__
    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)


class data_loader:
    """
    This class lazily loads training datasets (it makes debugging a lot faster if you
    don't have to wait for the entire dataset to load first).
    """
    def __init__(self, input_folder, image_size, label_size):
        self.images = {}
        self.labels = {}
        self.shrunk_width = image_size
        self.label_shrunk_width = label_size
        self.input_folder = input_folder

        self.input_image_names = list(map(lambda x: x.strip(), os.listdir(self.input_folder + "/images")))
        random.shuffle(self.input_image_names)
        self.clear_augmentation()

    def clear_augmentation(self):
        self.top_left_corner = [0, 0]
        self.bottom_right_corner = [1, 1]
        self.reflect_1 = 1
        self.reflect_2 = 1
        self.color_scaling = np.array([1, 1, 1])
        self.color_offset = np.array([0, 0, 0])

        self.images = {}
        self.labels = {}
    
    def update_data_augmentation(self):
        self.top_left_corner = np.random.randint(0, self.shrunk_width // 4, [2])
        self.bottom_right_corner = np.random.randint(1, self.shrunk_width // 4, [2])
        self.reflect_1, self.reflect_2 = np.random.randint(0, 1, [2]) * 2 - 1
        self.color_scaling = np.random.random([3]) * (1.2 - 0.8) + 0.8  # this produces random floats on the range [0.8, 1.2]
        self.color_offset = np.random.randint(-20, 20, [3])

        self.images = {}
        self.labels = {}

    def _augment_image(self, img, change_colors=False):
        """
        Data augmentation is a way of "stretching" limited datasets to get more useful training data out of them.
        The idea is to warp, rotate, and discolor training images so that every time the learning algorithm sees
        them they are a little different (and effectively somewhat new training data). This isn't as good as
        having a larger training dataset, but it helps. This function "augments" training images.
        """
        
        # for input images
        if len(img.shape) == 3:
            # crop the image a bit
            padded_shape = np.maximum(img.shape, [self.shrunk_width, self.shrunk_width, 3])
            padded = np.zeros(np.array(padded_shape) + np.array([1, 1, 0]))
            padded[:img.shape[0],:img.shape[1], :] =  img
            img_smaller = padded[self.top_left_corner[0] : -self.bottom_right_corner[0], self.top_left_corner[1] : -self.bottom_right_corner[1], :]
            # adjust the colors to simulate different lighting conditions
            color_distorted = np.minimum(255, np.maximum(img_smaller * self.color_scaling + self.color_offset, 0))
            # reflect the image
            return color_distorted[::self.reflect_1, ::self.reflect_2]
        
        # for label images
        elif len(img.shape) == 2:
            # crop the image a bit
            padded_shape = np.maximum(img.shape, [self.shrunk_width, self.shrunk_width])
            padded = np.zeros(np.array(padded_shape) + np.array([1, 1]))
            padded[:img.shape[0],:img.shape[1]] =  img
            img_smaller = padded[self.top_left_corner[0] : -self.bottom_right_corner[0], self.top_left_corner[1] : -self.bottom_right_corner[1]]
            # reflect the image. Don't distort colors on the label image
            return img_smaller[::self.reflect_1, ::self.reflect_2]
        else:
            raise Exception()
        

    def __getitem__(self, i):
        if i not in self.images.keys():
            unaugmented = self._read_img_from_disk(self.input_folder + "/images/" + self.input_image_names[i], cv2.IMREAD_COLOR)
            augmented = self._augment_image(unaugmented, change_colors=True)
            self.images[i] = cv2.resize(augmented, (self.shrunk_width, self.shrunk_width))[:, :, ::-1]
            
            label = self._read_img_from_disk(self.input_folder + "/gt/" + self.input_image_names[i], cv2.IMREAD_GRAYSCALE)
            intermediate = self._augment_image(label)
            label_resized = cv2.resize(intermediate, (self.label_shrunk_width, self.label_shrunk_width), interpolation=cv2.INTER_AREA).astype(np.float32)[:, :]
            self.labels[i] = (label_resized != 0).astype(np.float32) * 2 - 1
        
        return (self.images[i], self.labels[i])

    def get_path(self, index):
        return self.input_folder + "/images/" + self.input_image_names[index]

    def __len__(self):
        return len(self.input_image_names)

    @memoized
    def _read_img_from_disk(self, path, mode):
        """
        Saves a little time by not reading the same image from disk twice. The @memoized annotation means that if
        that every time this function is called it's inputs and outputs are saved. If it is called with the same 
        inputs a second timethen the outputs are looked up from the previous run and returned (instead of running 
        the function again).
        """
        return cv2.imread(path, mode)


def save_confusion_and_print_errors(confusion, model, test_images, network_name):
    # save the final confusion matrix
    class_names = [index_to_label_name[i] for i in range(2)]
    save_confusion_matrix(confusion, class_names, network_name+"_confusion.png")

    # # print some incorrectly labeled images:
    # print()
    # print("Misclassified test images:")
    # for i in range(len(test_images)):
    #     image, label = test_images[i]
    #     x = torch.from_numpy(image.flatten()[np.newaxis, ::]).float()
    #     label_index = label_name_to_index[label]
    #     predictions = model(x).data.cpu().numpy().squeeze()

    #     if predictions.argmax() != label_index:
    #         # this image was misclassified. First sort the attempted predictions
    #         predictions_sorted = np.stack((np.linspace(0, 4, 5), predictions)).transpose()
    #         predictions_sorted = predictions_sorted[predictions_sorted[:, 1].argsort()].transpose()
    #         # now print out the predictions in order from most likely to least likely
    #         predicted_labels = [index_to_label_name[i] for i in predictions_sorted[0]]
    #         print("Image " + test_images.get_path(i) + " was predicted as ", predicted_labels,
    #               "(in order from highest probability to lowest)")