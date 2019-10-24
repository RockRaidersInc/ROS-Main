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

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 8, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 1, 3, padding=1)

        self.conv_single = torch.nn.Conv2d(3, 1, 3, padding=1)

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
        x4 = self.pool(self.nonlinear(self.conv4(x2)))
        output = self.tanh(x4)
        return output
        # return self.tanh(self.conv_single(x_prepared))
        # return x_raw[:, 0, :, :]


def main():
    train_images = data_loader("train", shrunk_width, shrunk_width/8)
    test_images = data_loader("test", shrunk_width, shrunk_width/8)

    # train_images = data_loader("train", shrunk_width, shrunk_width)
    # test_images = data_loader("test", shrunk_width, shrunk_width)

    epochs = 1000  # number of times to go through the training set

    N = 2  # batch size

    model = ConvNet()

    loss_fn = torch.nn.L1Loss()

    learning_rate = 1e-1
    # use the ADAM optimizer because it has fewer parameters to tune than SGD
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch_num in range(epochs):
        model.train()  # turn on dropout
        epoch_correct_predictions = 0
        print()
        for i in range(0, len(train_images), N):
            print("\ron image " + str(i), end='')

            # Forward pass: compute predicted y by passing x to the model.
            next_batch_images = []
            next_batch_labels = []
            for j in range(N):
                image, label = train_images[(i + j) % len(train_images)]
                # next_batch_images.append(np.swapaxes(image, 0, 2))
                next_batch_labels.append(label)

            # batch = torch.from_numpy(np.stack(next_batch_images)).float()
            batch = prepare_images_for_nn([train_images[(i + j) % len(train_images)][0] for j in range(N)])
            label_images = torch.from_numpy(np.array(next_batch_labels)).float()

            predictions = model(batch)
            # predictions = batch[:, 0, :, :]

            # Compute and print loss.
            pixelwise_loss = torch.pow(predictions - label_images, 2)
            loss = torch.mean((label_images + 1) * pixelwise_loss * 100 + pixelwise_loss)
            print(" \t error: ", loss.data.cpu().numpy())


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("\r", end="")
        # print("average epoch training accuracy:", epoch_correct_predictions / len(train_images) / np.product(train_images[0][0].shape))


        # # now that the network has been trained for one epoch, test it on the training data
        # model.eval()  # turn off dropout
        # confusion = np.zeros((2, 2), dtype=np.int32)
        # for i in range(len(train_images)):
        #     image, label = train_images[i]
        #     x = prepare_images_for_nn([image])

        #     prediction = (model(x).data.cpu().numpy() > 0).astype(np.int32)
        #     label_thresholded = (label > 0).astype(np.int32)

        #     for x in range(prediction.shape[2]):
        #         for y in range(prediction.shape[3]):
        #             confusion[label_thresholded[0, x, y], prediction[0, 0, x, y]] += 1

        # print("confusion matrix after epoch", epoch_num)
        # print("row is correct label, column is predicted label")
        # for i in range(2):
        #     print(" " * (11 - len(index_to_label_name[i])) + index_to_label_name[i] + ": ", end="")
        #     print(confusion[i])

        # n_correct_predictions = np.sum(confusion * np.eye(2, 2)) / np.product(label.shape)
        # print("training accuracy: %3.4f%%" % (n_correct_predictions / len(train_images) * 100,))



        # now that the network has been trained for one epoch, test it on the testing data
        model.eval()  # turn off dropout
        confusion = np.zeros((2, 2), dtype=np.int32)
        for i in range(len(test_images)):
            image, label = test_images[i]
            x = prepare_images_for_nn([image])

            prediction_raw = model(x).data.cpu().numpy().astype(np.float32)
            prediction = (prediction_raw > 0).astype(np.int32)
            label_thresholded = (label > 0).astype(np.int32)

            f, (ax1, ax2) = plt.subplots(1, 2)

            ax1.imshow(image.squeeze())

            resized = cv2.resize((np.dstack([prediction_raw.squeeze(), prediction_raw.squeeze(), prediction_raw.squeeze()]) * 127 + 127).astype(np.uint8), 
                                                        dsize=(shrunk_width, shrunk_width), interpolation=cv2.INTER_CUBIC)

            # ax2.imshow(image.squeeze())
            ax2.imshow(resized[:, :, 0] / 127, cmap='jet', alpha=1)

            if epoch_num % 1 == 0:
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
        print("testing accuracy: %3.4f%%" % (n_correct_predictions / len(test_images) * 100,))

        optimizer.zero_grad()  # just to make sure it doesn't learn from the testing data

    # save the final confusion matrix and print out misclassified images
    save_confusion_and_print_errors(confusion, model, test_images, "convolutional_network")


def prepare_images_for_nn(imgs):
    return torch.from_numpy(np.stack([np.swapaxes(image, 0, 2).swapaxes(1, 2) for image in imgs])).float()



if __name__ == "__main__":
    main()