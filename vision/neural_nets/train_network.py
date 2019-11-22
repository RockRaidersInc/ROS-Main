from __future__ import print_function

import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import PIL

import math

from nn_utils import *
from confusion_mat_tools import save_confusion_matrix
import networks


np.set_printoptions(precision=6)

def whopskip():
    print("\r\n plopper" * 100)


# make numpy printing more readable
int_formatter = lambda x: "%3if" % x
np.set_printoptions(formatter={'int_kind':int_formatter})
np.set_printoptions(precision=5, suppress=True)


input_file_list = "image_list.txt"
shrunk_width = 416


def main():
    train_images = data_loader("train", shrunk_width, shrunk_width)
    test_images = data_loader("test", shrunk_width, shrunk_width)

    epochs = 1000  # number of times to go through the training set
    N = 1  # batch size
    # model = networks.Yolo2Transfer(pretrained_weights="nn_weights").to(get_default_torch_device())
    model = networks.Yolo2Transfer().to(get_default_torch_device())

    loss_fn = torch.nn.L1Loss()

    learning_rate = 1e-4
    # use the ADAM optimizer because it has fewer parameters to tune than SGD
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    best_test_accuracy = 0

    for epoch_num in range(epochs):
        model.train()  # turn on dropout
        epoch_correct_predictions = 0
        print()
        loss_list = []

        print("starting epoch " + str(epoch_num))
        for i in range(0, len(train_images), N):
            print("\ron image " + str(i) + "/" + str(len(train_images)), end='')

            optimizer.zero_grad()  # just to make sure the network doesn't learn from any previous testing or training data

            # Forward pass: compute predicted y by passing x to the model.
            next_batch_images_unprocessed = []
            next_batch_labels = []
            next_batch_exclusion_mask = []
            for j in range(N):
                image, label, excluded = train_images[(i + j) % len(train_images)]
                next_batch_images_unprocessed.append(image)
                next_batch_labels.append(np.array(label))
                next_batch_exclusion_mask.append(np.array(excluded))

            label_images = torch.from_numpy(np.array(next_batch_labels)).float()

            if label_images.max() == -1:
                # there are no lane points in this batch
                continue

            exclusion_images = torch.from_numpy(np.array(next_batch_exclusion_mask)).float()
            predictions = model(next_batch_images_unprocessed).to("cpu")
            # lane_bg_scaling_factor = 1 / (np.mean(np.array(next_batch_labels) / 2 + 0.5) + 0.0001) + 0.75

            # Compute and print loss.
            # pixelwise_squared_diff = torch.pow(predictions - label_images, 2)
            # lane_pixel_loss = (label_images + 1) / 2 * pixelwise_squared_diff * lane_bg_scaling_factor    # component of loss from pixels on a lane
            # other_pixel_loss = (label_images - 1) / -2 * pixelwise_squared_diff        # component of loss from pixels in the background
            # loss = torch.mean((lane_pixel_loss + other_pixel_loss) * (1 - exclusion_images))

            # Compute and print loss.
            # lane_pixel_loss = np.abs((label_images + 1) / 2 * lane_bg_scaling_factor - 1)  # image with 0 where there is no lane, higher where there is lane
            # pixelwise_squared_diff = torch.pow((predictions - label_images) * lane_pixel_loss, 2)
            # loss = torch.mean(pixelwise_squared_diff)

            positive_points = (label_images * (1 - exclusion_images) + 1) / 2  # one where there is a lane, 0 otherwise
            negative_points = (label_images * (1 - exclusion_images) - 1) / -2  # one where there is a lane, 0 otherwise

            squared_loss = torch.pow(predictions - label_images, 2)
            true_positives = torch.sum(positive_points * squared_loss)
            true_negatives = torch.sum(negative_points * squared_loss)
            loss = (true_positives / (true_positives + torch.sum(positive_points) + 0.00001) + 
                    (true_negatives / (true_negatives + torch.sum(negative_points) + 0.00001))) / 2  # 0.00001 added to avoid division by 0

            # loss = torch.mean(torch.pow(predictions - label_images, 2))
            # print(" \t loss: ", loss.data.cpu().numpy())
            loss_list.append(loss.data.cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_images.update_data_augmentation()
            
        train_images.shuffle()
        print()
        print("loss from epoch " + str(epoch_num) + ": " + str(np.mean(loss_list)))
        # print("average epoch training accuracy:", epoch_correct_predictions / len(train_images) / np.product(train_images[0][0].shape))

        # now that the network has been trained for one epoch, test it on the testing data
        if epoch_num % 4 == 3:
            evaluate_model(model, train_images, epoch_num, train=True, show_images=False, sample=0.2)
            epoch_test_accuracy = evaluate_model(model, test_images, epoch_num, train=False, show_images=True if epoch_num % 4 == 3 else False)
            
            if best_test_accuracy < epoch_test_accuracy:
                best_test_accuracy = epoch_test_accuracy
                torch.save(model.state_dict(), "nn_weights_epoch_" + str(epoch_num))

    # save the final confusion matrix and print out misclassified images
    # save_confusion_and_print_errors(confusion, model, test_images, "convolutional_network")


def evaluate_model(model, image_set, epoch_num, train=False, show_images=False, sample=1):
    model.eval()  # turn off dropout
    image_set.clear_augmentation()
    image_set.shuffle()
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
        f, ax_list = plt.subplots(6, 3)
        f.set_size_inches(10,10)
        displayed_accuracies = []
        for i, ax in enumerate(ax_list):
            image, label, _ = image_set[all_accuracies_images[(i - 2) % len(all_accuracies_images)][1]]
            displayed_accuracies.append(all_accuracies_images[(i - 2) % len(all_accuracies_images)][0])
            ax[0].imshow(image.squeeze().astype(np.uint8))

            x = [image]
            prediction_raw = np.tanh(model(x).data.cpu().numpy().astype(np.float32))
            # resized = cv2.resize((np.dstack([prediction_raw.squeeze(), prediction_raw.squeeze(), prediction_raw.squeeze()]) * 127 + 127).astype(np.uint8), 
            #                                             dsize=(shrunk_width, shrunk_width), interpolation=cv2.INTER_CUBIC)
            # resized[0, 0, 0] = 127
            # resized[0, 1, 0] = -127
            # ax[1].imshow(resized[:, :, 0] / 127, cmap='jet', alpha=1)
            ax[1].imshow(prediction_raw.squeeze(), cmap='jet', alpha=1)

            label[0, 0] = 1
            label[0, 1] = -1
            ax[2].imshow(label, cmap='jet', alpha=1)
        plt.suptitle("worst results from testing, balanced accuracies are: \n" + ", ".join(map(lambda x: str(x)[:5], displayed_accuracies)))
        plt.savefig("output_epoch_" + str(epoch_num) + ".png", dpi=100)
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
    prediction = (prediction_raw > 0).astype(np.int32)
    label_thresholded = (label > 0).astype(np.int32)

    for x in range(prediction.shape[2]):
        for y in range(prediction.shape[3]):
            confusion[label_thresholded[x, y], prediction[0, 0, x, y]] += 1
    return confusion

if __name__ == "__main__":
    main()
