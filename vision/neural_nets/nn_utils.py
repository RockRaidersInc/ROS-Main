import pdb
import random
import numpy as np
import cv2
import torch
import os
import functools
import collections
from PIL import Image

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


def get_default_torch_device():
    """ Pick GPU if available, otherwise use CPU """
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


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


# from https://codereview.stackexchange.com/questions/153029/recursively-list-files-within-a-directory
def files_within(directory_path):
    pass
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for file_name in filenames:
            if file_name.lower().endswith("jpg") or file_name.lower().endswith("png"):
                relative_path = os.path.join(dirpath[len(directory_path): ], file_name)
                yield relative_path if relative_path[0] != "/" else relative_path[1:]



class data_loader:
    """
    This class lazily loads training datasets (it makes debugging a lot faster if you
    don't have to wait for the entire dataset to load first).
    """
    def __init__(self, input_folder, image_size, label_size, num_augmentation_sets=5):
        self.unaugmented_images = {}
        self.unaugmented_labels = {}
        self.unaugmented_excluded = {}
        self.augmented_image_sets = [{} for i in range(num_augmentation_sets)]
        self.augmented_label_sets = [{} for i in range(num_augmentation_sets)]
        self.augmented_excluded_sets = [{} for i in range(num_augmentation_sets)]
        self.shrunk_width = image_size
        self.label_shrunk_width = label_size
        self.input_folder = os.path.join("../combined_dataset", input_folder)

        self.input_image_names = list([x for x in files_within(os.path.join(self.input_folder, "images")) if "no" not in x])
        self.image_ordering = list(range(len(self.input_image_names)))

        self.augmentation_specs = [{"top_left_corner": np.random.randint(0, self.shrunk_width // 4, [2]),
                                   "bottom_right_corner": np.random.randint(1, self.shrunk_width // 4, [2]),
                                   "reflect_1": 1,  # don't actually fip long vertical axis, that wouldn't ever represent real world data
                                   "reflect_2": np.random.randint(0, 2, [1])[0] * 2 - 1,
                                   "color_scaling": (np.random.random([3]) * (1.1 - 0.9) + 0.9),
                                   "color_offset": np.random.randint(-10, 10, [3])} for i in range(num_augmentation_sets)]
        self.augmentation_current_index = 0

        self.clear_augmentation()

    def clear_augmentation(self):
        
        self.top_left_corner = [0, 0]
        self.bottom_right_corner = [1, 1]
        self.reflect_1 = 1
        self.reflect_2 = 1
        self.color_scaling = np.array([1, 1, 1])
        self.color_offset = np.array([0, 0, 0])
        
        self.images = self.unaugmented_images
        self.labels = self.unaugmented_labels
        self.excluded = self.unaugmented_excluded

    def shuffle(self):
        random.shuffle(self.image_ordering)

    def unshuffle(self):
        self.image_ordering.sort()

    def update_data_augmentation(self):
        """
        self.top_left_corner = np.random.randint(0, self.shrunk_width // 4, [2])
        self.bottom_right_corner = np.random.randint(1, self.shrunk_width // 4, [2])
        self.reflect_1, self.reflect_2 = np.random.randint(0, 1, [2]) * 2 - 1
        self.color_scaling = np.random.random([3]) * (1.2 - 0.8) + 0.8  # this produces random floats on the range [0.8, 1.2]
        self.color_scaling *= np.random.random([1]) * 0.5 + 0.5  # simulate low lighting where all colors are equally effected
        self.color_offset = np.random.randint(-20, 20, [3])
        """

        image_set_num = self.augmentation_current_index
        self.augmentation_current_index = (self.augmentation_current_index + 1) % len(self.augmentation_specs)

        self.top_left_corner = self.augmentation_specs[image_set_num]["top_left_corner"]
        self.bottom_right_corner = self.augmentation_specs[image_set_num]["bottom_right_corner"]
        self.reflect_1 = self.augmentation_specs[image_set_num]["reflect_1"]
        self.reflect_2 = self.augmentation_specs[image_set_num]["reflect_2"]
        self.color_scaling = self.augmentation_specs[image_set_num]["color_scaling"]
        self.color_offset = self.augmentation_specs[image_set_num]["color_offset"]

        self.images = self.augmented_image_sets[image_set_num]
        self.labels = self.augmented_label_sets[image_set_num]
        self.excluded = self.augmented_excluded_sets[image_set_num]

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
        

    def __getitem__(self, i_raw):
        i = self.image_ordering[i_raw]
        if i not in self.images.keys():
            unaugmented = self._read_img_from_disk(self.input_folder + "/images/" + self.input_image_names[i], cv2.IMREAD_COLOR)
            augmented = self._augment_image(unaugmented, change_colors=True)
            image_resized = cv2.resize(augmented, (self.shrunk_width, self.shrunk_width), interpolation=cv2.INTER_AREA != 0).astype(np.uint8)
            self.images[i] = image_resized
            
            raw_label = self._read_img_from_disk(self.input_folder + "/gt/" + self.input_image_names[i], cv2.IMREAD_COLOR)
            label = self._augment_image(raw_label)

            excluded = ((label[:, :, 0] == 255) * (label[:, :, 1] != 255)).astype(np.uint8)
            excluded_resized = cv2.resize(excluded, (self.label_shrunk_width, self.label_shrunk_width), interpolation=cv2.INTER_AREA != 0).astype(np.float32)[:, :]
            self.excluded[i] = excluded_resized
            
            intermediate = (label[:, :, 1] > 70).astype(np.uint8)
            label_opened = cv2.morphologyEx((intermediate != 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5)))
            label_resized = cv2.resize(label_opened, (self.label_shrunk_width, self.label_shrunk_width), interpolation=cv2.INTER_AREA != 0).astype(np.float32)[:, :]
            self.labels[i] = (label_resized).astype(np.float32) * 2 - 1

        return (self.images[i], self.labels[i], self.excluded[i])

    def get_path(self, index):
        return self.input_folder + "/images/" + self.input_image_names[index]

    def __len__(self):
        return len(self.input_image_names)

    def _read_img_from_disk(self, path, mode):
        """
        Saves a little time by not reading the same image from disk twice. The @memoized annotation means that if
        that every time this function is called it's inputs and outputs are saved. If it is called with the same 
        inputs a second timethen the outputs are looked up from the previous run and returned (instead of running 
        the function again).
        """
        if mode == cv2.IMREAD_COLOR:
            return np.array(Image.open(path).convert('RGB'))
        else:
            return cv2.imread(path, mode)


def prepare_images_for_nn(imgs):
    """
    Turn an array of images into a tensor and move it to the GPU if the GPU is being used.
    imgs is a list of 3d arrays (width x height x 3)
    """
    raise Exception("deprecated")
    return torch.from_numpy(np.stack([image for image in imgs])).transpose(1,2).transpose(1,3).float().div(255.0).to(get_default_torch_device())


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
