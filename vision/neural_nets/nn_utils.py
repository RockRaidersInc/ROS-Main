import math
import pdb
import random
import numpy as np
import cv2
import torch
import os
import functools
import collections
from PIL import Image
import unittest
from sortedcontainers import SortedDict

# from confusion_mat_tools import save_confusion_matrix

np.random.seed(0)

# label_name_to_onehot = {"other" :     np.array([1, 0]),     "lane":      np.array([0, 1])}

# label_name_to_index = {"other" :     0,    "lane":     1}

index_to_label_name = {0: "other", 1: "lane"}


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
                relative_path = os.path.join(dirpath[len(directory_path):], file_name)
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

        if int(image_size) != image_size or int(label_size) != label_size:
            raise Error("image_size and label_size must be integers")

        self.shrunk_width = int(image_size)
        self.label_shrunk_width = int(label_size)
        self.input_folder = os.path.join("../combined_dataset", input_folder)

        self.input_image_names = list(
            [x for x in files_within(os.path.join(self.input_folder, "images")) if "no" not in x])
        random.shuffle(self.input_image_names)

        self.image_ordering = list(range(len(self)))

        def gen_cloud_mask(center, radius, darkness):
            xx, yy = np.meshgrid(np.linspace(0, 1, self.shrunk_width), np.linspace(0, 1, self.shrunk_width))
            radei_from_center = np.sqrt(np.power(xx - center[0], 2) + np.power(yy - center[1], 2))
            lighting_values = np.maximum(radius - radei_from_center,
                                         np.zeros(radei_from_center.shape)) / radius * darkness
            return 1 - lighting_values

        self.augmentation_specs = [{"top_left_corner": np.random.randint(0, self.shrunk_width // 4, [2]),
                                    "bottom_right_corner": np.random.randint(1, self.shrunk_width // 4, [2]),
                                    "reflect_1": 1,
                                    # don't actually fip long vertical axis, that wouldn't ever represent real world data
                                    "reflect_2": np.random.randint(0, 2, [1])[0] * 2 - 1,
                                    "color_scaling": (np.random.random([3]) * (1.2 - 0.8) + 0.8),
                                    "color_offset": np.random.randint(-20, 20, [3]),
                                    "cloud_mask": gen_cloud_mask(np.random.random([2]),
                                                                 np.random.random([1])[0] * 2 + 0.1,
                                                                 np.random.random([1])[0] * 0.5 + 0.1),
                                    "scale": np.random.random([1])[0] * 0.3 + 1.0,
                                    "rotate": (np.random.random([1])[0] * 2 - 1) * 30
                                    } for i in range(num_augmentation_sets)]
        self.augmentation_current_index = 0

        self.clear_augmentation()
        self.positive_sums = np.zeros(len(self), dtype=np.float64)
        self.negative_sums = np.zeros(len(self), dtype=np.float64)

    def get_postive_negative_ratio(self):
        """returns the ratio of positive pixels to negative pixels in the entire dataset
        (Note: this function will return very large numbers (on the order of 10^19) if there are no negative
        points in the dataset but will not return nan)"""

        return (self.positive_sums.sum()) / (self.negative_sums.sum() + 10 ** (-18))

    def clear_augmentation(self):
        """

        """
        self.top_left_corner = [0, 0]
        self.bottom_right_corner = [1, 1]
        self.reflect_1 = 1
        self.reflect_2 = 1
        self.color_scaling = np.array([1, 1, 1])
        self.color_offset = np.array([0, 0, 0])
        self.cloud_mask = np.ones([self.shrunk_width, self.shrunk_width])
        self.scale = 1.
        self.rotate = 0.

        self.images = self.unaugmented_images
        self.labels = self.unaugmented_labels
        self.excluded = self.unaugmented_excluded

    def shuffle(self):
        random.shuffle(self.image_ordering)

    def unshuffle(self):
        self.image_ordering.sort()

    def update_data_augmentation(self, index=None):
        """
        """

        if len(self.augmentation_specs) == 0:
            return  # there are no augmented images so there is nothing to do

        if index is None:
            image_set_num = self.augmentation_current_index
            self.augmentation_current_index = (self.augmentation_current_index + 1) % len(self.augmentation_specs)
        else:
            # an index of zero means the unaugmented images
            if index == 0:
                self.clear_augmentation()
                return
            else:
                image_set_num = index - 1
                self.augmentation_current_index = index - 1

        self.top_left_corner = self.augmentation_specs[image_set_num]["top_left_corner"]
        self.bottom_right_corner = self.augmentation_specs[image_set_num]["bottom_right_corner"]
        self.reflect_1 = self.augmentation_specs[image_set_num]["reflect_1"]
        self.reflect_2 = self.augmentation_specs[image_set_num]["reflect_2"]
        self.color_scaling = self.augmentation_specs[image_set_num]["color_scaling"]
        self.color_offset = self.augmentation_specs[image_set_num]["color_offset"]
        self.cloud_mask = self.augmentation_specs[image_set_num]["cloud_mask"]
        self.scale = self.augmentation_specs[image_set_num]["scale"]
        self.rotate = self.augmentation_specs[image_set_num]["rotate"]

        self.images = self.augmented_image_sets[image_set_num]
        self.labels = self.augmented_label_sets[image_set_num]
        self.excluded = self.augmented_excluded_sets[image_set_num]

    def _augment_image(self, img_unrotated, change_colors=False):
        """
        Data augmentation is a way of "stretching" limited datasets to get more useful training data out of them.
        The idea is to warp, rotate, and discolor training images so that every time the learning algorithm sees
        them they are a little different (and effectively somewhat new training data). This isn't as good as
        having a larger training dataset, but it helps. This function "augments" training images.
        """

        resized = img_unrotated
        center = (resized.shape[1] / 2, resized.shape[0] / 2)
        M = cv2.getRotationMatrix2D(center, self.rotate, self.scale)
        img = cv2.warpAffine(resized, M, resized.shape[:2][::-1])

        # for input images
        if len(img.shape) == 3:
            # crop the image a bit
            padded_shape = np.maximum(img.shape, [self.shrunk_width, self.shrunk_width, 3])
            padded = np.zeros(np.array(padded_shape) + np.array([1, 1, 0]))
            padded[:img.shape[0], :img.shape[1], :] = img
            img_smaller = padded[self.top_left_corner[0]: -self.bottom_right_corner[0],
                          self.top_left_corner[1]: -self.bottom_right_corner[1], :]

            # adjust the colors to simulate different lighting conditions
            color_distorted = np.minimum(255, np.maximum(img_smaller * self.color_scaling + self.color_offset, 0))
            # reflect the image
            return color_distorted[::self.reflect_1, ::self.reflect_2]

        # for label images
        elif len(img.shape) == 2:
            # crop the image a bit
            padded_shape = np.maximum(img.shape, [self.shrunk_width, self.shrunk_width])
            padded = np.zeros(np.array(padded_shape) + np.array([1, 1]))
            padded[:img.shape[0], :img.shape[1]] = img
            img_smaller = padded[self.top_left_corner[0]: -self.bottom_right_corner[0],
                          self.top_left_corner[1]: -self.bottom_right_corner[1]]
            # reflect the image. Don't distort colors on the label image
            return img_smaller[::self.reflect_1, ::self.reflect_2]
        else:
            raise Exception()

    def __getitem__(self, i_raw):

        # TODO: remove
        # i_raw = i_raw % 10


        i_reordered = self.image_ordering[i_raw]

        # change the augmentation set
        self.update_data_augmentation(index=int(i_reordered) // int(len(self.input_image_names)))
        i = i_reordered % len(self.input_image_names)

        if i not in self.images.keys():
            unaugmented = self._read_img_from_disk(self.input_folder + "/images/" + self.input_image_names[i],
                                                   cv2.IMREAD_COLOR)
            augmented = self._augment_image(unaugmented, change_colors=True)
            image_resized = cv2.resize(augmented, (self.shrunk_width, self.shrunk_width),
                                       interpolation=cv2.INTER_AREA != 0).astype(np.uint8)
            cloud_applied = (image_resized * self.cloud_mask[:, :, np.newaxis]).astype(np.uint8)
            self.images[i] = cloud_applied

            raw_label = self._read_img_from_disk(self.input_folder + "/gt/" + self.input_image_names[i],
                                                 cv2.IMREAD_COLOR)
            label = self._augment_image(raw_label)

            excluded = ((label[:, :, 0] == 255) * (label[:, :, 1] != 255)).astype(np.uint8)
            excluded_resized = cv2.resize(excluded, (self.label_shrunk_width, self.label_shrunk_width),
                                          interpolation=cv2.INTER_AREA != 0).astype(np.float32)[:, :]
            self.excluded[i] = excluded_resized

            intermediate = (label[:, :, 1] > 70).astype(np.uint8)

            # label_opened = cv2.morphologyEx((intermediate != 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5)))
            label_opened = (intermediate != 0).astype(np.uint8)

            # dilate the label image to counteract lane pixels disappearing after shrinking
            label_dilated = cv2.dilate(label_opened,
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * 3 + 1, 2 * 3 + 1), (3, 3)))

            label_resized = cv2.resize(label_dilated, (self.label_shrunk_width, self.label_shrunk_width),
                                       interpolation=cv2.INTER_AREA != 0).astype(np.float32)[:, :]
            self.labels[i] = (label_resized).astype(np.float32) * 2 - 1

            self.positive_sums[i_raw] = np.sum(self.labels[i] == 1)
            self.negative_sums[i_raw] = np.sum(self.labels[i] == -1)

        return (self.images[i], self.labels[i], self.excluded[i])

    def get_path(self, index):
        return self.input_folder + "/images/" + self.input_image_names[index]

    def __len__(self):
        return len(self.input_image_names) * (1 + len(self.augmented_image_sets))

    def _read_img_from_disk(self, path, mode):
        if mode == cv2.IMREAD_COLOR:
            return np.array(Image.open(path).convert('RGB'))
        else:
            return cv2.imread(path, mode)


class HardNegativeMiningManager:
    """
    The idea behind hard negative mining is that training points with the greatest error (hard negatives)
    are trained on more often than examples with less error. This class keeps track of the training error
    of images in a dataset and makes it easy to pull out the image with the lowest error. It also has an
    option to require that images only go so many training iterations between sightings (so every image is
    trained on at least once every n epochs).
    """

    def __init__(self, image_set_size, max_iterations_between_sightings, min_iters_between_sightings=-1):
        """
        NOTE: max_iterations_between_sightings is expressed in images, not epochs (to set the maximum
        number of epochs use image_set_size * max_epochs_between_sightings as max_iterations_between_sightings).
        """
        self.set_size = image_set_size
        self.max_iters_between_sightings = max_iterations_between_sightings
        self.min_iters_between_sightings = min_iters_between_sightings

        self.losses = [np.nan] * self.set_size
        self.losses_dict = SortedDict()  # stores values as (loss, {images})
        self.last_iter_seen = np.zeros([self.set_size], dtype=np.float64) - (max_iterations_between_sightings + 1)

        # store unseen images as a set instead of a list for performance reasons
        self.nan_images = {index for index in range(image_set_size)}

    def update_loss(self, current_iter, index, loss):
        assert not np.isnan(loss)
        loss = float(-loss)
        # find and remove the passed image form losses_dict
        old_loss = self.losses[index]
        if not np.isnan(old_loss):
            assert old_loss in self.losses_dict
            assert index in self.losses_dict[old_loss]
            self.losses_dict[old_loss].remove(index)

            # no sense in keeping a loss around if it has no associated images
            if len(self.losses_dict[old_loss]) == 0:
                del self.losses_dict[old_loss]


        # now add the image back to losses_dict
        if index in self.nan_images:
            self.nan_images.remove(index)
        self.losses_dict.setdefault(loss, default=set()).add(index)
        self.losses[index] = loss

        self.last_iter_seen[index] = current_iter

    def get_next(self, current_iter):
        return self.get_n_next(current_iter, 1)[0]

    def get_n_next(self, current_iter, n):
        def check_repeats(lst):
            if len(lst) != len({i for i in lst}):
                print()
                print("train image batch list has repeats")

        if n > len(self.last_iter_seen):
            raise Exception("more images requested from get_n_next() than are in the training set")

        # first add any images which have a loss of nan, this signifies that they have not been trained on yet
        ret_indexes = [index for index in self.nan_images]
        ret_indexes = ret_indexes[:n]
        check_repeats(ret_indexes)
        if len(ret_indexes) == n:
            return ret_indexes

        # which image hasn't been trained on for the most iterations? Immediately take ones that are too old
        image_ages = current_iter - self.last_iter_seen
        ret_indexes += [index for index, age in sorted(((index, age) for index, age in enumerate(image_ages)if
                        (age >= self.max_iters_between_sightings and index not in ret_indexes)), key=lambda x: x[1])]
        ret_indexes = ret_indexes[:n]  # don't take more than n images
        check_repeats(ret_indexes)
        if len(ret_indexes) == n:
            return ret_indexes

        # now find the images with the highest training error that arent already being returned
        ret_indexes += [index for loss, index in get_indexes(self.losses_dict)
                        if (image_ages[index] >= self.min_iters_between_sightings
                            and index not in ret_indexes)]
        ret_indexes = ret_indexes[:n]
        check_repeats(ret_indexes)
        if len(ret_indexes) == n:
            return ret_indexes

        # Ths probably means that self.min_iters_between_sightings is larger than the number of training images. Just add images
        # by their previous loss ignoring how recently they've been trained on.
        ret_indexes += [index for loss, index in get_indexes(self.losses_dict)
                        if index not in ret_indexes]
        ret_indexes = ret_indexes[:n]
        check_repeats(ret_indexes)
        if len(ret_indexes) == n:
            return ret_indexes

        # The function should have returned by now, there is probably a bug if control flow gets here
        print()
        print("error in get_n_next(), not enough images found")
        pdb.set_trace()

    # def get_n_next_old(self, current_iter, n):
    #     losses = np.copy(self.losses)
    #     last_iter_seen = np.copy(self.last_iter_seen)
    #     ret_indexes = []
    #     for i in range(n):
    #         next_index = self.get_next(current_iter + i)
    #         ret_indexes.append(next_index)
    #         self.update_loss(current_iter + i, next_index,
    #                          -np.inf)  # give this image a low loss so we won't get it returned again
    #         if i > 0 and ret_indexes[i] == ret_indexes[i - 1]:
    #             pdb.set_trace()
    #     self.losses = losses
    #     self.last_iter_seen = last_iter_seen
    #     return ret_indexes

    def max_loss(self):
        return np.nanmax(self.losses)

    def average_loss(self):
        return np.nanmean(self.losses)


def get_indexes(x: dict):
    """
    Accepts a dictionary where all values are iterable and returns key, value pairs for
    each item in every stored value
    ex: {4:[a, b], 5:{c, d}} would yield (4, a), (4, b), (5, c), (5, d)
    :return:
    """
    for key in x:
        for i in x[key]:
            yield (key, i)



class TestHardwareNegativeManager(unittest.TestCase):
    def test_get_next_and_update_loss(self):
        manager = HardNegativeMiningManager(4, 1000)
        manager.update_loss(0, 0, 1)
        manager.update_loss(1, 1, 1.1)
        manager.update_loss(2, 2, 2)
        self.assertEqual(manager.get_next(3), 3)
        manager.update_loss(3, 3, 1)
        self.assertEqual(manager.get_next(4), 2)
        manager.update_loss(4, 2, 0)
        self.assertEqual(manager.get_next(5), 1)

    def test_max_iters_between_sightings(self):
        manager = HardNegativeMiningManager(2, 5)
        manager.update_loss(0, 0, 1)
        self.assertEqual(manager.get_next(1), 1)
        # 1
        manager.update_loss(1, 1, 2)
        self.assertEqual(manager.get_next(2), 1)
        # 2
        manager.update_loss(2, 1, 2)
        self.assertEqual(manager.get_next(3), 1)
        # 3
        manager.update_loss(3, 1, 2)
        self.assertEqual(manager.get_next(4), 1)
        # 4
        manager.update_loss(4, 1, 2)
        self.assertEqual(manager.get_next(5), 0)
        # 5
        manager.update_loss(5, 0, 1)
        self.assertEqual(manager.get_next(6), 1)
        # 6
        manager.update_loss(6, 1, 2)
        self.assertEqual(manager.get_next(7), 1)
        # 7
        manager.update_loss(7, 1, 2)
        self.assertEqual(manager.get_next(8), 1)
        # 8
        manager.update_loss(8, 1, 2)
        self.assertEqual(manager.get_next(9), 1)
        # 9
        manager.update_loss(9, 1, 2)
        self.assertEqual(manager.get_next(10), 0)
        # 10
        manager.update_loss(10, 1, 2)
        self.assertEqual(manager.get_next(11), 0)

    def test_max_loss(self):
        manager = HardNegativeMiningManager(4, 4)
        manager.update_loss(0, 0, 1)
        self.assertEqual(manager.max_loss(), 1)
        manager.update_loss(1, 1, 0.5)
        self.assertEqual(manager.max_loss(), 1)
        manager.update_loss(2, 2, 4)
        self.assertEqual(manager.max_loss(), 4)
        manager.update_loss(3, 1, 5)
        self.assertEqual(manager.max_loss(), 5)

    def test_average_loss(self):
        manager = HardNegativeMiningManager(4, 4)
        manager.update_loss(0, 0, 1)
        self.assertEqual(manager.average_loss(), 1.)
        manager.update_loss(1, 1, 0.5)
        self.assertEqual(manager.average_loss(), 0.75)
        manager.update_loss(2, 2, 1.5)
        self.assertEqual(manager.average_loss(), 1.)
        manager.update_loss(3, 1, 2)
        self.assertEqual(manager.average_loss(), 1.5)
