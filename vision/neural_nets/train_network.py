from __future__ import print_function
import math
import time;

import pdb

import cProfile

import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import PIL

from nn_utils import *
from confusion_mat_tools import save_confusion_matrix
import networks


# make numpy printing more readable
np.set_printoptions(formatter={'int_kind':lambda x: "%3if" % x})
np.set_printoptions(precision=5, suppress=True)

hard_negative_mining = False

input_file_list = "image_list.txt"
shrunk_width = int(416*1.5)


def main():
    train_images = data_loader("train", shrunk_width, shrunk_width)
    test_images = data_loader("test", shrunk_width, shrunk_width)

    epochs = 5000  # number of times to go through the training set
    batch_size = 1  # batch size

    model = networks.Yolo2Transfer().to(get_default_torch_device())
    # model = networks.Yolo2Transfer_smaller().to(get_default_torch_device())

    learning_rate = 1e-4
    # use the ADAM optimizer because it has fewer parameters to tune than SGD
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0000)  # weight decay was 0.0001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    best_test_accuracy = 0

    for epoch_num in range(epochs):
        model.train()  # turn on dropout
        epoch_correct_predictions = 0
        print()
        loss_list = []

        data_loading_time = 0
        inference_time = 0
        backprop_update_time = 0

        print("starting epoch " + str(epoch_num))
        # pdb.set_trace()
        for i in range(0, len(train_images), batch_size):
            print("\ron image " + str(i) + "/" + str(len(train_images)), end='')

            optimizer.zero_grad()  # just to make sure the network doesn't learn from any previous testing or training data
            last_time = time.time()

            last_time = time.time()

            # Forward pass: compute predicted y by passing x to the model.
            next_batch_images_unprocessed = []
            next_batch_labels = []
            next_batch_exclusion_mask = []
            for j in range(batch_size):
                image, label, excluded = train_images[(i + j) % len(train_images)]
                next_batch_images_unprocessed.append(image)
                next_batch_labels.append(np.array(label))
                next_batch_exclusion_mask.append(np.array(excluded))

            label_images = torch.from_numpy(np.array(next_batch_labels)).float()

            exclusion_images = torch.from_numpy(np.array(next_batch_exclusion_mask)).float()

            data_loading_time += time.time() - last_time
            last_time = time.time()

            predictions = model(next_batch_images_unprocessed).to("cpu")

            inference_time += time.time() - last_time
            last_time = time.time()

            positive_points = (label_images ==  1) * (exclusion_images == 0).type(torch.float32)
            negative_points = (label_images == -1) * (exclusion_images == 0).type(torch.float32)

            squared_loss = torch.pow(predictions - label_images, 2)
            true_positives = torch.sum(positive_points * squared_loss)
            true_negatives = torch.sum(negative_points * squared_loss)

            loss = (true_positives / (torch.sum(positive_points) + 0.001) +
                    (true_negatives / (torch.sum(negative_points) + 0.001))) / 2  # 0.001 added to avoid division by 0

            # loss = torch.mean(torch.pow(predictions - label_images, 2))
            # print(" \t loss: ", loss.data.cpu().numpy())
            loss_cpu = loss.data.cpu().numpy()
            loss_list.append(loss_cpu)

            # for the hard negative mining, only compute gradients and update the weights on the worst quarter of training images.
            # the gradient is always updated with the first 12 images because there isn't a good estimate of the 50th precentile loss
            # until a few images from the current batch have been trained on. (it's somewhat arbitrary)
            if not hard_negative_mining or (len(loss_list) < 13 or sorted(loss_list)[int(len(loss_list) // 2)] <= loss_cpu):
                loss.backward()
                optimizer.step()

            backprop_update_time += time.time() - last_time
            last_time = time.time()
            train_images.update_data_augmentation()

        # how much of a speedup do we get from not updating the augmentation?
        if epoch_num % 1 == 0:
            train_images.shuffle()

        print()
        print("loss from epoch " + str(epoch_num) + ": " + str(np.mean(loss_list)))
        print("average time spent loading image: %2.3f, running model: %2.3f, backprop and weight updates: %2.3f" % 
                (data_loading_time / len(train_images), inference_time / len(train_images), backprop_update_time / len(train_images)))

        train_images.shuffle()

        # now that the network has been trained for one epoch, test it on the testing data
        if epoch_num == 5 or epoch_num % 10 == 0:
            evaluate_model(model, train_images, epoch_num, train=True, show_images=True)
            epoch_test_accuracy = evaluate_model(model, test_images, epoch_num, train=False, show_images=True)
            # epoch_test_accuracy = evaluate_model(model, train_images, epoch_num, train=True, show_images=True if epoch_num % 5 == 0 else False)
            if best_test_accuracy < epoch_test_accuracy:
                best_test_accuracy = epoch_test_accuracy
                torch.save(model.state_dict(), "nn_weights_epoch_" + str(epoch_num))

    # save the final confusion matrix and print out misclassified images
    # save_confusion_and_print_errors(confusion, model, test_images, "convolutional_network")


def evaluate_model(model, image_set, epoch_num, train=False, show_images=False, sample=1):
    model.eval()  # turn off dropout (if the network uses dropout)
    image_set.clear_augmentation()
    # image_set.shuffle()
    confusion = np.zeros((2, 2), dtype=np.float64)
    image_accuracies = []
    all_accuracies_images = []

    for i in range(int(math.floor(len(image_set) * sample))):
        image, label, excluded = image_set[i]
        new_confusion = get_confusion_mat_single_image(model, image, label)
        balanced_accuracy = get_balanced_accuracy(new_confusion)
        all_accuracies_images.append((balanced_accuracy, i))
        confusion += new_confusion

    print("confusion matrix after epoch", epoch_num)
    print("row is correct label, column is predicted label")
    for i in range(2):
        print(" " * (11 - len(index_to_label_name[i])) + index_to_label_name[i] + ": ", end="")
        print(confusion[i] / confusion.sum())

    image_accuracies.append(np.sum(confusion * np.eye(2, 2)) / np.sum(confusion))
    if train:
        print("training balanced accuracy: ", end="")
    else:
        print("testing balanced accuracy: ", end="")
    print("%3.4f%%" % (get_balanced_accuracy(confusion) * 100))
    average_accuracy = float(np.mean(list(map(lambda x: float(x[0]), all_accuracies_images))) * 100)
    print("average balanced accuracy from all images: %3.4f%%" % average_accuracy)

    all_accuracies_images.sort(key=lambda x: x[0])

    if show_images:
        f, ax_list = plt.subplots(3, 6)
        f.set_size_inches(15,10)
        displayed_accuracies = []
        for i in range(len(ax_list[0])):
            image, label, _ = image_set[all_accuracies_images[(i - 2) % len(all_accuracies_images)][1]]
            displayed_accuracies.append(all_accuracies_images[(i - 2) % len(all_accuracies_images)][0])
            ax_list[0][i].imshow(image.squeeze().astype(np.uint8))

            x = [image]
            prediction_raw = np.tanh(model(x).data.cpu().numpy().astype(np.float32))
            # resized = cv2.resize((np.dstack([prediction_raw.squeeze(), prediction_raw.squeeze(), prediction_raw.squeeze()]) * 127 + 127).astype(np.uint8), 
            #                                             dsize=(shrunk_width, shrunk_width), interpolation=cv2.INTER_CUBIC)
            # resized[0, 0, 0] = 127
            # resized[0, 1, 0] = -127
            # ax[1].imshow(resized[:, :, 0] / 127, cmap='jet', alpha=1)
            ax_list[1][i].imshow(prediction_raw.squeeze(), cmap='jet', alpha=1)

            label[0, 0] = 1
            label[0, 1] = -1
            ax_list[2][i].imshow(label, cmap='jet', alpha=1)
        plt.suptitle("best and worst results from testing, balanced accuracies are: \n" + ", ".join(map(lambda x: str(x)[:5], displayed_accuracies)))
        plt.savefig("output_epoch_" + str(epoch_num) + ("_train" if train else "_test") + ".png", dpi=100)
        plt.close(f)

    return average_accuracy


def get_balanced_accuracy(confusion):
    """
    confusion should be in the form 
    [[true positive,  false negative]
     [false positive, true negative ]]
    """
    return float(((confusion[0, 0] / (confusion[0, 0] + confusion[0, 1] + 0.0000001)) + 
                (confusion[1, 1] / (confusion[1, 0] + confusion[1, 1] + 0.0000001))) / 2)


def get_confusion_mat_single_image(model, image, label):
    confusion = np.zeros((2, 2), dtype=np.float64)
    x = [image]

    prediction_raw = model(x).data.cpu().numpy().astype(np.float32)
    prediction = (prediction_raw > 0)
    label_thresholded = (label > 0)

    true_positives = np.count_nonzero(label_thresholded * prediction)
    true_negatives = np.count_nonzero(~label_thresholded * ~prediction)
    false_positives = np.count_nonzero(~label_thresholded * prediction)
    false_negatives = np.count_nonzero(label_thresholded * ~prediction)

    confusion[0, 0] = true_negatives
    confusion[1, 0] = false_negatives
    confusion[0, 1] = false_positives
    confusion[1, 1] = true_positives

    return confusion

if __name__ == "__main__":
    cProfile.run('main()', 'profile_results')
