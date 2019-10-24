import random
import numpy as np
import cv2
import torch
import os

from confusion_mat_tools import save_confusion_matrix


label_name_to_onehot = {"other" :     np.array([1, 0]),
                        "lane":      np.array([0, 1])}

label_name_to_index = {"other" :     0,
                        "lane":     1}

index_to_label_name = {0: "other",
                       1: "lane"}

def to_indices(arr):
    return np.array(list(map(lambda ar: ar.argmax(), [arr[i] for i in range(arr.shape[0])])))


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

    def __getitem__(self, i):
        if i not in self.images.keys():
            image = cv2.imread(self.input_folder + "/images/" + self.input_image_names[i], cv2.IMREAD_COLOR)
            self.images[i] = cv2.resize(image, (self.shrunk_width, self.shrunk_width))[:, :, ::-1]
            
            label = cv2.imread(self.input_folder + "/gt/" + self.input_image_names[i], cv2.IMREAD_GRAYSCALE)
            label_resized = cv2.resize(label, (self.label_shrunk_width, self.label_shrunk_width)).astype(np.float32)[:, :] / 128 - 1
            self.labels[i] = (label_resized > 0.8).astype(np.float32)
        
        return (self.images[i], self.labels[i])

    def get_path(self, index):
        return self.input_folder + "/images/" + self.input_image_names[index]

    def __len__(self):
        return len(self.input_image_names)


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