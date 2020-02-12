from __future__ import print_function;
import math
import time
import random

import pdb

import cProfile

import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors, cm
import PIL

from nn_utils import *
from confusion_mat_tools import save_confusion_matrix
import networks


# make numpy printing more readable
np.set_printoptions(formatter={'int_kind':lambda x: "%3if" % x})
np.set_printoptions(precision=5, suppress=True)

hard_negative_mining = False

shrunk_width = int(416*1.5)


def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3, training_log={}):

    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    def lr_lambda(it):
        # next_lr = min_lr + (max_lr - min_lr) * relative(it, stepsize)
        epoch = training_log["current_epoch"]
        next_lr = max_lr * 10 ** (-math.floor(epoch/400.))
        # training_log["lr_iter"].append(epoch)
        # training_log["lr_val"].append(next_lr)
        return next_lr

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda


def main():
    train_images = data_loader("train", shrunk_width, shrunk_width)
    test_images = data_loader("test", shrunk_width, shrunk_width)

    epochs = 5000  # number of times to go through the training set
    batch_size = 4  # batch size

    model = networks.Yolo2Transfer().to(get_default_torch_device())
    # model = networks.Yolo2Transfer_smaller().to(get_default_torch_device())

    initial_learning_rate = 1e-3
    momentum = 0.9
    weight_decay = 0.000
    optimizer_name = "SGD"

    training_log = {"training_loss_iter": [],
                    "training_loss_val": [],
                    "training_balanced_acc_iter": [],
                    "training_balanced_acc_val": [],
                    "testing_balanced_acc_iter": [],
                    "testing_balanced_acc_val": [],
                    "learning_rate_iter": [],
                    "learning_rate_val": [],
                    "lr_iter": [],
                    "lr_val": [],
                    "optimizer": optimizer_name,
                    "initial_learning_rate": initial_learning_rate,
                    "momentum": momentum,
                    "batch_size": batch_size,
                    "current_epoch": 0}

    # use the ADAM optimizer because it has fewer parameters to tune than SGD
    if optimizer_name == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=weight_decay)  # weight decay was 0.0001
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=momentum, weight_decay=weight_decay)
        step_size = 5 * len(train_images)
        clr = cyclical_lr(step_size, min_lr=1e-4, max_lr=1e-0, training_log=training_log)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])

    best_test_accuracy = 0

    for epoch_num in range(epochs):
        training_log["current_epoch"] = epoch_num
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

            if random.random() < 0.5:
                train_images.update_data_augmentation()

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

            predictions = torch.nn.Tanh()(predictions * 1.1)

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
                scheduler.step()  # this updates the learning rate

            backprop_update_time += time.time() - last_time
            last_time = time.time()
            train_images.update_data_augmentation()

        print()
        print("loss from epoch " + str(epoch_num) + ": " + str(np.mean(loss_list)))
        print("average time spent loading image: %2.3f, running model: %2.3f, backprop and weight updates: %2.3f" %
              (data_loading_time / len(train_images), inference_time / len(train_images),
               backprop_update_time / len(train_images)))

        # how much of a speedup do we get from not updating the augmentation?
        if epoch_num % 1 == 0:
            train_images.shuffle()

        # save the epoch loss so it can be graphed later
        training_log["training_loss_iter"].append(epoch_num)
        training_log["training_loss_val"].append(np.mean(loss_list))
        training_log["lr_iter"].append(epoch_num)
        training_log["lr_val"].append(scheduler.get_lr()[0])


        # re-randomize the order training images are viewed in
        train_images.shuffle()

        # now that the network has been trained for one epoch, test it on the testing data
        if epoch_num == 5 or epoch_num % 10 == 0:
            epoch_train_accuracy = evaluate_model(model, train_images, epoch_num, train=True, show_images=True)
            epoch_test_accuracy = evaluate_model(model, test_images, epoch_num, train=False, show_images=True)
            # epoch_test_accuracy = evaluate_model(model, train_images, epoch_num, train=True, show_images=True if epoch_num % 5 == 0 else False)

            # if this was the best iteration so far then save the weights
            if best_test_accuracy < epoch_test_accuracy:
                best_test_accuracy = epoch_test_accuracy
                torch.save(model.state_dict(), "nn_weights_epoch_" + str(epoch_num))

            # add data to training log so graphs of it can be made
            training_log["training_balanced_acc_iter"].append(epoch_num)
            training_log["training_balanced_acc_val"].append(epoch_train_accuracy)
            training_log["testing_balanced_acc_iter"].append(epoch_num)
            training_log["testing_balanced_acc_val"].append(epoch_test_accuracy)

            # actually write the graphs
            save_graphs(training_log)

    # save the final confusion matrix and print out misclassified images
    # save_confusion_and_print_errors(confusion, model, test_images, "convolutional_network")


def save_graphs(training_log):
    """
    Saves a graph with a bunch of data about the network training (loss, test accuracy, learning rate, ect)
    """
    f = plt.figure()
    ax1 = f.add_subplot(111)
    f.set_size_inches(15,10)
    ax2 = ax1.twinx()
    lr_ax = ax1.twinx()

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("balanced accuracy (out of 100%)")
    ax2.set_ylabel("training loss")
    lr_ax.set_ylabel("learning rate")
    lr_ax.set_yscale('log')

    max_iter = max(training_log["training_loss_iter"])

    line_1 = ax2.plot(training_log["training_loss_iter"], training_log["training_loss_val"], c="b", label="training loss")
    line_2 = ax1.plot(training_log["training_balanced_acc_iter"], training_log["training_balanced_acc_val"], c="r", label="training accuracy")
    line_3 = ax1.plot(training_log["testing_balanced_acc_iter"], training_log["testing_balanced_acc_val"], c="g", label="testing accuracy")
    line_4 = lr_ax.plot(training_log["lr_iter"], training_log["lr_val"], c="m", label="learning rate")

    lines = line_1 + line_2 + line_3 + line_4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower left")

    title_settings_string = ("Optimizer: " + training_log["optimizer"] +
                             ", initial learning rate: " + str(training_log["initial_learning_rate"]) +
                             ", momentum: " + str(training_log["momentum"]) +
                             ", batch size: " + str(training_log["batch_size"]))

    lr_ax.spines['right'].set_position(('outward', 60))

    plt.suptitle("Training history after " + str(max_iter) + " iterations\n" + title_settings_string)
    plt.savefig("training_graphs.png", dpi=100)
    plt.close(f)



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

    colorbar_norm = colors.Normalize(vmin=-1, vmax=1, clip=True)

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
            im_obj = ax_list[1][i].imshow(prediction_raw.squeeze(), cmap='jet', alpha=1, norm=colorbar_norm)
            f.colorbar(im_obj, ax=ax_list[1][i])
            print(model(x).data.cpu().numpy().max(), model(x).data.cpu().numpy().min(), model(x).data.cpu().numpy().mean())

            label[0, 0] = 1
            label[0, 1] = -1
            gt_im_obj = ax_list[2][i].imshow(label, cmap='jet', alpha=1, norm=colorbar_norm)
            f.colorbar(gt_im_obj, ax=ax_list[2][i])

        plt.suptitle("best and worst results from " + ("training" if train else "testing") + ", balanced accuracies are: \n" + ", ".join(map(lambda x: str(x)[:5], displayed_accuracies)))
        f.tight_layout()
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
    main()
    # cProfile.run('main()', 'profile_results')
