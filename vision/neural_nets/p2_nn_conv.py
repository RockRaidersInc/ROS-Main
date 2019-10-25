from __future__ import print_function

import torch
from matplotlib import pyplot as plt

from nn_utils import *
from confusion_mat_tools import save_confusion_matrix


# make numpy printing more readable
int_formatter = lambda x: "%3if" % x
np.set_printoptions(formatter={'int_kind':int_formatter})
np.set_printoptions(precision=5, suppress=True)


input_file_list = "image_list.txt"
shrunk_width = 512


class SmallerNet(torch.nn.Module):
    """
    This network appears to generalize quite well, even when only training with the small igvc_sim_trainset_3 dataset
    (test error is within 3% of training error). It doesn't work super well though.
    """
    def __init__(self):
        super(SmallerNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 12, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(12, 16, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 1, 3, padding=1)

        self.pool = torch.nn.MaxPool2d(2, 2)
        self.nonlinear = torch.nn.functional.elu
        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh
        self.dropout = torch.nn.Dropout(0.4)  # drop 50% of the neurons
    
    def forward(self, x_raw):
        x_prepared = x_raw / 128 - 1
        x1 = self.pool(self.nonlinear(self.conv1(x_prepared)))
        x2 = self.pool(self.nonlinear(self.conv2(x1)))
        x3 = self.pool(self.nonlinear(self.conv3(x2)))
        x4 = self.nonlinear(self.conv4(x3))
        output = self.tanh(x4)  # using tanh to get a value between -1 and 1
        return output


class LargerNet(torch.nn.Module):
    def __init__(self):
        super(LargerNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.conv2_1 = torch.nn.Conv2d(8, 12, 3, padding=1)
        self.conv2_2 = torch.nn.Conv2d(12, 12, 1, padding=0)
        self.conv2_3 = torch.nn.Conv2d(12, 12, 3, padding=1)

        self.conv3_1 = torch.nn.Conv2d(12, 16, 3, padding=1)
        self.conv3_2 = torch.nn.Conv2d(16, 32, 1, padding=0)
        self.conv3_3 = torch.nn.Conv2d(32, 16, 3, padding=1)

        self.conv4_1 = torch.nn.Conv2d(16, 36, 3, padding=1)
        self.conv4_2 = torch.nn.Conv2d(36, 36, 1, padding=0)
        self.conv4_3 = torch.nn.Conv2d(36, 1, 3, padding=1)

        self.pool = torch.nn.MaxPool2d(2, 2)
        self.nonlinear = torch.nn.functional.elu
        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh
        self.dropout = torch.nn.Dropout(0.4)  # drop 50% of the neurons
    
    def forward(self, x_raw):
        x_prepared = x_raw / 128 - 1
        x1 = self.pool(self.nonlinear(self.conv1(x_prepared)))
        x2_1 = self.nonlinear(self.conv2_1(x1))
        x2_2 = self.nonlinear(self.conv2_2(x2_1))
        x2_3 = self.pool(self.nonlinear(self.conv2_3(x2_2)))
        x3_1 = self.nonlinear(self.conv3_1(x2_3))
        x3_2 = self.nonlinear(self.conv3_2(x3_1))
        x3_3 = self.pool(self.nonlinear(self.conv3_3(x3_2)))
        x4_1 = self.nonlinear(self.conv4_1(x3_3))
        x4_2 = self.nonlinear(self.conv4_2(x4_1))
        x4_3 = self.nonlinear(self.conv4_3(x4_2))
        output = self.tanh(x4_3)  # using tanh to get a value between -1 and 1
        return output


def main():
    train_images = data_loader("train", shrunk_width, shrunk_width/8)
    test_images = data_loader("test", shrunk_width, shrunk_width/8)

    # train_images = data_loader("train", shrunk_width, shrunk_width)
    # test_images = data_loader("test", shrunk_width, shrunk_width)

    epochs = 1000  # number of times to go through the training set

    N = 2  # batch size

    model = SmallerNet()
    # model = LargerNet()

    loss_fn = torch.nn.L1Loss()

    learning_rate = 1e-2
    # use the ADAM optimizer because it has fewer parameters to tune than SGD
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch_num in range(epochs):
        model.train()  # turn on dropout
        epoch_correct_predictions = 0
        print()
        for i in range(0, len(train_images), N):
            print("\ron image " + str(i), end='')

            # Forward pass: compute predicted y by passing x to the model.
            next_batch_images_unprocessed = []
            next_batch_labels = []
            for j in range(N):
                image, label = train_images[(i + j) % len(train_images)]
                next_batch_images_unprocessed.append(image)
                next_batch_labels.append(label)

            # batch = torch.from_numpy(np.stack(next_batch_images)).float()
            batch = prepare_images_for_nn(next_batch_images_unprocessed)
            label_images = torch.from_numpy(np.array(next_batch_labels)).float()

            predictions = model(batch)

            lane_bg_scaling_factor = 1 / (np.mean(np.array(next_batch_labels) / 2 + 0.5) + 0.0001) + 0.75
            # print("   ", lane_bg_scaling_factor, (np.array(next_batch_labels) / 2 + 0.5).sum(), np.product(np.array(next_batch_labels).shape))

            # Compute and print loss.
            pixelwise_squared_diff = torch.pow(predictions - label_images, 2)
            lane_pixel_loss = (label_images + 1) / 2 * pixelwise_squared_diff * lane_bg_scaling_factor    # component of loss from pixels on a lane
            other_pixel_loss = (label_images - 1) / -2 * pixelwise_squared_diff        # component of loss from pixels in the background
            loss = torch.mean(lane_pixel_loss + other_pixel_loss)
            # print(" \t loss: ", loss.data.cpu().numpy())


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_images.update_data_augmentation()

        print("\r", end="")
        # print("average epoch training accuracy:", epoch_correct_predictions / len(train_images) / np.product(train_images[0][0].shape))

        # now that the network has been trained for one epoch, test it on the testing data
        evaluate_model(model, train_images, epoch_num, train=True, show_images=True if epoch_num % 30 == 29 else False)
        evaluate_model(model, test_images, epoch_num, train=False, show_images=True if epoch_num % 30 == 29 else False)

        optimizer.zero_grad()  # just to make sure it doesn't learn from the testing data

    # save the final confusion matrix and print out misclassified images
    # save_confusion_and_print_errors(confusion, model, test_images, "convolutional_network")


def evaluate_model(model, image_set, epoch_num, train=False, show_images=False):
    model.eval()  # turn off dropout
    # image_set.clear_augmentation()
    confusion = np.zeros((2, 2), dtype=np.int32)
    for i in range(len(image_set)):
        image, label = image_set[i]
        x = prepare_images_for_nn([image])

        prediction_raw = model(x).data.cpu().numpy().astype(np.float32)
        prediction = (prediction_raw > 0).astype(np.int32)
        label_thresholded = (label > 0).astype(np.int32)

        if show_images:
            f, (ax1, ax2, ax3) = plt.subplots(1, 3)

            ax1.imshow(image.squeeze().astype(np.uint8))

            resized = cv2.resize((np.dstack([prediction_raw.squeeze(), prediction_raw.squeeze(), prediction_raw.squeeze()]) * 127 + 127).astype(np.uint8), 
                                                        dsize=(shrunk_width, shrunk_width), interpolation=cv2.INTER_CUBIC)

            # ax2.imshow(image.squeeze())
            resized[0, 0, 0] = 127
            resized[0, 1, 0] = -127
            ax2.imshow(resized[:, :, 0] / 127, cmap='jet', alpha=1)

            label[0, 0] = 1
            label[0, 1] = -1
            ax3.imshow(label, cmap='jet', alpha=1)
            plt.show()
            plt.close(f)

        for x in range(prediction.shape[2]):
            for y in range(prediction.shape[3]):
                confusion[label_thresholded[x, y], prediction[0, 0, x, y]] += 1
    
    print("confusion matrix after epoch", epoch_num)
    print("row is correct label, column is predicted label")
    for i in range(2):
        print(" " * (11 - len(index_to_label_name[i])) + index_to_label_name[i] + ": ", end="")
        print(confusion[i])

    n_correct_predictions = np.sum(confusion * np.eye(2, 2)) / np.product(label.shape)
    if train:
        print("training accuracy:")
    else:
        print("testing accuracy:")
    print("%3.4f%%" % (n_correct_predictions / len(image_set) * 100,))



def prepare_images_for_nn(imgs):
    return torch.from_numpy(np.stack([np.swapaxes(image, 0, 2).swapaxes(1, 2) for image in imgs])).float()



if __name__ == "__main__":
    main()