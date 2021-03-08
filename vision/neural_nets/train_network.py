from __future__ import print_function
import math
import time
import random
import os

import pdb

import cProfile

global use_matplotlib
use_matplotlib = True

import torch
from torch import nn

if use_matplotlib:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib import colors, cm
import PIL

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
    from torch2trt import TRTModule
    from torch2trt import torch2trt
except ImportError:
    raise ImportError("Please install apex or torch2trt")

from nn_utils import *
import networks
import networks2

import unittest


# make numpy printing more readable
np.set_printoptions(formatter={'int_kind':lambda x: "%3if" % x})
np.set_printoptions(formatter={'float_kind':lambda x: "%3.7f" % x})
np.set_printoptions(precision=5, suppress=True)


# variables that won't change during a training run
image_width_input = int(416)
image_width_output = image_width_input / 4
balanced_loss_horizontal_scale = 5.0
balanced_loss_linearity_factor = 0.1


def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3, training_log={}):

    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    def lr_lambda(it):
        # next_lr = min_lr + (max_lr - min_lr) * relative(it, stepsize)
        # epoch = training_log["current_epoch"]
        # next_lr = max_lr * 10 ** (-math.floor(epoch/240.))
        next_lr = max_lr * 10 ** (-math.floor(it / (10 * 1000.)))
        return next_lr

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda


def main():
    train_images = data_loader("train", image_width_input, image_width_output, num_augmentation_sets=20)
    test_images = data_loader("test", image_width_input, image_width_output, num_augmentation_sets=0)

    epochs = 700  # number of times to go through the training set
    batch_size = 16  # batch size

    # model = networks.Yolo2Transfer().to(get_default_torch_device())
    # model = networks.Yolo2TransferSmaller().to(get_default_torch_device())
    # model = networks.Yolo2TransferDilated().to(get_default_torch_device())
    # model = networks.Yolo2TransferUndilated().to(get_default_torch_device())
    # model = networks.Yolo2TransferPixelnet().to(get_default_torch_device())
    # model = networks.Yolo2TransferPixelnetMoreYolo(init_yolo_weights=False).to(get_default_torch_device())
    # model = networks.PixelnetNoYolo(init_yolo_weights=False).to(get_default_torch_device())
    model = networks2.PixelMobileNet().to(get_default_torch_device())

    initial_learning_rate = 1e-1
    momentum = 0.9
    weight_decay = 0.0001
    optimizer_name = "SGD"
    hard_negative_mining = True
    epoch_size = 1000
    hard_negative_revisit_period = int(2 * len(train_images))

    # make a directory for any files produced by this training run
    try:
        next_training_folder_num = max([int(''.join(j for j in i if j.isdigit())) for i in os.listdir(".") if "training_outputs_" in i] + [-1]) + 1
        output_dir = "training_outputs_" + str(next_training_folder_num) + "/"
        os.makedirs(output_dir)
    except OSError:
        raise

    # This stores a bunch of information about the current training run
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
                    "hard_negative_mining": hard_negative_mining,
                    "hard_negative_revisit_period": hard_negative_revisit_period,
                    "current_epoch": 0,
                    "output_dir": output_dir,
                    "image_width_input": image_width_input,
                    "image_width_output": image_width_output,
                    "balanced_loss_horizontal_scale": balanced_loss_horizontal_scale,
                    "balanced_loss_linearity_factor": balanced_loss_linearity_factor,
                    "network_class_name": model.__class__.__name__}

    # write the contents of the training log to a file (so the settings of this training run are recorded)
    outfile = open(training_log["output_dir"] + "settings.txt", 'w')
    outfile.write(str(sorted([str(i) + ": " + str(training_log[i]) for i in training_log.keys()])).replace(",", ",\n") + "\n")
    outfile.close()

    # set up the optimizer
    if optimizer_name == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=weight_decay)  # weight decay was 0.0001
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=momentum, weight_decay=weight_decay)
        step_size = 5 * len(train_images)
        clr = cyclical_lr(step_size, min_lr=1e-4, max_lr=1e-0, training_log=training_log)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])

    negative_mining_manager = HardNegativeMiningManager(len(train_images),
                                                        hard_negative_revisit_period if hard_negative_mining else 1,
                                                        min_iters_between_sightings=200)

    # model, optimizer = amp.initialize(model, optimizer,
    #                                   opt_level="O0",
    #                                   keep_batchnorm_fp32=True
    #                                   )

    best_test_accuracy = 0

    for epoch_num in range(0, epochs):
        training_log["current_epoch"] = epoch_num

        model.train()  # turn on dropout and batch norm

        epoch_correct_predictions = 0
        print()
        loss_list = []

        data_loading_time = 0
        inference_time = 0
        backprop_update_time = 0

        print("starting epoch " + str(epoch_num))

        # model.load_state_dict(torch.load("save_nn_weights_epoch_0"))

        for i in range(0, epoch_size, batch_size):
            print("\ron image {:-4}/{:}".format(i, epoch_size), end='')

            optimizer.zero_grad()  # just to make sure the network doesn't learn from any previous testing or training data
            last_time = time.time()

            # train_images.unshuffle()

            # Forward pass: compute predicted y by passing x to the model.
            next_batch_images_unprocessed = []
            next_batch_labels = []
            next_batch_exclusion_mask = []
            batch_image_indexes = negative_mining_manager.get_n_next(epoch_num * epoch_size + i, batch_size)
            print(batch_image_indexes, end='')
            for index in batch_image_indexes:
                image, label, excluded = train_images[index % len(train_images)]
                next_batch_images_unprocessed.append(image)
                next_batch_labels.append(np.array(label))
                next_batch_exclusion_mask.append(np.array(excluded))

            label_images = torch.from_numpy(np.array(next_batch_labels)).float().to(get_default_torch_device())
            exclusion_images = torch.from_numpy(np.array(next_batch_exclusion_mask)).float().to(get_default_torch_device())

            data_loading_time += time.time() - last_time
            last_time = time.time()

            # predict the training images
            predictions = model.do_images(next_batch_images_unprocessed)
            inference_time += time.time() - last_time
            last_time = time.time()

            # loss_by_image = 1 - approx_balanced_accuracy(predictions, label_images, exclusion_images)
            loss_by_image = 1 - IoU(predictions, label_images, exclusion_images)
            loss = torch.mean(loss_by_image)

            loss_cpu = loss.data.cpu().numpy()
            loss_list.append(loss_cpu)
            print(", ", loss_cpu, end='')

            loss_by_image_cpu = loss_by_image.data.cpu().numpy().reshape([-1])
            for j in range(len(batch_image_indexes)):
                negative_mining_manager.update_loss(epoch_num * epoch_size + i + j, batch_image_indexes[j], loss_by_image_cpu[j])

            loss.backward()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()

            optimizer.step()
            if optimizer_name == "SGD":
                scheduler.step()  # this updates the learning rate

            backprop_update_time += time.time() - last_time
            last_time = time.time()
            train_images.update_data_augmentation()

        print()
        print("loss from epoch " + str(epoch_num) + ": " + str(np.mean(loss_list)))
        print("average time: loading data: %2.4f, running model: %2.4f, weight updates: %2.4f" %
              (data_loading_time / epoch_size, inference_time / epoch_size, backprop_update_time / epoch_size))

        # save the epoch loss so it can be graphed later
        training_log["training_loss_iter"].append(epoch_num)
        # training_log["training_loss_val"].append(np.mean(loss_list))
        training_log["training_loss_val"].append(negative_mining_manager.average_loss())
        training_log["lr_iter"].append(epoch_num)
        if optimizer_name == "SGD":
            training_log["lr_val"].append(scheduler.get_last_lr()[0])
        else:
            training_log["lr_val"].append(0)

        #if optimizer_name == "SGD":
        #    optimizer = torch.optim.SGD(model.parameters(), lr=initial_learning_rate * 2 ** (-math.floor(epoch_num/40.)), momentum=momentum, weight_decay=weight_decay)
        #    step_size = 5 * len(train_images)
        #    clr = cyclical_lr(step_size, min_lr=1e-4, max_lr=1e-0, training_log=training_log)
        #    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])

        # now that the network has been trained for one epoch, test it on the testing data
        if epoch_num % 4 == 0:
            # print("no TRT")
            # epoch_train_accuracy = evaluate_model(model, train_images, epoch_num, train=True, show_images=True, training_log=training_log, sample=0.2, useTRT=False)
            # print("basic TRT")
            # epoch_train_accuracy = evaluate_model(model, train_images, epoch_num, train=True, show_images=True, training_log=training_log, sample=0.2)
            print("fp16")
            epoch_train_accuracy = evaluate_model(model, train_images, epoch_num, train=True, show_images=True,
                                                  training_log=training_log, sample=0.2, fp16=True)
            # print("int8")
            # epoch_train_accuracy = evaluate_model(model, train_images, epoch_num, train=True, show_images=True,
            #                                       training_log=training_log, sample=0.2, int8=True)
            epoch_test_accuracy = evaluate_model(model, test_images, epoch_num, train=False, show_images=True, training_log=training_log, useTRT=True)
            # epoch_test_accuracy = evaluate_model(model, train_images, epoch_num, train=True, show_images=True if epoch_num % 5 == 0 else False)

            # if this was the best iteration so far then save the weights
            if best_test_accuracy < epoch_test_accuracy:
                best_test_accuracy = epoch_test_accuracy
                torch.save(model.state_dict(), training_log["output_dir"] + "nn_weights_epoch_" + str(epoch_num))

            # add data to training log so graphs of it can be made
            training_log["training_balanced_acc_iter"].append(epoch_num)
            training_log["training_balanced_acc_val"].append(epoch_train_accuracy)
            training_log["testing_balanced_acc_iter"].append(epoch_num)
            training_log["testing_balanced_acc_val"].append(epoch_test_accuracy)

            # actually write the graphs
            save_graphs(training_log)

def save_graphs(training_log):
    """
    Saves a graph with a bunch of data about the network training (loss, test accuracy, learning rate, ect)
    """
    include_lr = training_log["optimizer"] == "SGD"

    if not use_matplotlib:
        return

    f = plt.figure()
    ax1 = f.add_subplot(111)
    f.set_size_inches(15,10)
    ax2 = ax1.twinx()
    if include_lr: lr_ax = ax1.twinx()

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("IoU (out of 100%)")
    ax1.set_ylim([0, 100])
    ax1.yaxis.set_ticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax2.set_ylabel("Training Loss")
    if include_lr:
        lr_ax.set_ylabel("Learning Rate")
        lr_ax.set_yscale('log')

    max_iter = max(training_log["training_loss_iter"])

    line_1 = ax2.plot(training_log["training_loss_iter"], training_log["training_loss_val"], c="b", label="training loss")
    line_2 = ax1.plot(training_log["training_balanced_acc_iter"], training_log["training_balanced_acc_val"], c="r", label="training accuracy")
    line_3 = ax1.plot(training_log["testing_balanced_acc_iter"], training_log["testing_balanced_acc_val"], c="g", label="testing accuracy")
    if include_lr:
        line_4 = lr_ax.plot(training_log["lr_iter"], training_log["lr_val"], c="m", label="learning rate")

    lines = line_1 + line_2 + line_3 + line_4 if include_lr else line_1 + line_2 + line_3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower left")

    title_settings_string = ("Optimizer: " + training_log["optimizer"] +
                             ", Initial Learning Rate: " + str(training_log["initial_learning_rate"]) +
                             ", Momentum: " + str(training_log["momentum"]) +
                             ", Batch Size: " + str(training_log["batch_size"]))

    if include_lr:
        lr_ax.spines['right'].set_position(('outward', 60))

    ax1.grid()

    plt.suptitle("Training history after " + str(max_iter) + " Iterations\n" + title_settings_string)
    plt.savefig(training_log["output_dir"] + "training_graphs.png", dpi=100)
    plt.close(f)


def evaluate_model(model, image_set, epoch_num, train=False, show_images=False, sample=0.3, training_log=None, useTRT=True, fp16=False, int8=False):
    model.eval()  # turn off dropout and batch norm (if the network uses dropout)
    # image_set.clear_augmentation()
    image_set.unshuffle()
    confusion = np.zeros((2, 2), dtype=np.float64)
    all_accuracies_images = []

    if useTRT:
        input = model.prep_images([image_set[0][0]])
        model_trt = torch2trt(model, [input], fp16_mode=fp16, int8_mode=int8)
        # no_trt_out = model(input)
        # trt_out = model_trt(input)
        # print("max deviation between trt and non_trt models: " + str(torch.max(torch.abs(trt_out - no_trt_out))))

        inputs = []
        outputs = []
        for i in (int(i) for i in np.linspace(0, len(image_set) - 1, int(len(image_set) * sample))):
            image, label, excluded = image_set[i]
            x = model.prep_images([image])
            inputs.append((i, x, label, excluded))

        start_time = time.time()
        for i, x, label, excluded in inputs:
            prediction_raw = model_trt(x).data.cpu().numpy().astype(np.float32)
            outputs.append((i, prediction_raw, label, excluded))
        print("average model inference and accuracy calculation time: %3.1f ms" % (
        (time.time() - start_time) * 1000 / (len(image_set) * sample)))

        for i, prediction_raw, label, excluded in outputs:
            new_confusion = get_confusion_mat_single_image(prediction_raw, label)
            balanced_accuracy = get_balanced_accuracy(new_confusion)
            IoU = get_IoU_single_image(prediction_raw, label)
            all_accuracies_images.append((IoU, i))
            confusion += new_confusion

    else:
        inputs = []
        outputs = []
        for i in (int(i) for i in np.linspace(0, len(image_set) - 1, int(len(image_set) * sample))):
            image, label, excluded = image_set[i]
            x = model.prep_images([image])
            inputs.append((i, x, label, excluded))

        start_time = time.time()
        for i, x, label, excluded in inputs:
            prediction_raw = model(x).data.cpu().numpy().astype(np.float32)
            outputs.append((i, prediction_raw, label, excluded))
        print("average model inference and accuracy calculation time: %3.1f ms" % (
        (time.time() - start_time) * 1000 / (len(image_set) * sample)))

        for i, prediction_raw, label, excluded in outputs:
            new_confusion = get_confusion_mat_single_image(prediction_raw, label)
            balanced_accuracy = get_balanced_accuracy(new_confusion)
            IoU = get_IoU_single_image(prediction_raw, label)
            all_accuracies_images.append((IoU, i))
            confusion += new_confusion
        model_trt = None


    print("confusion matrix after epoch", epoch_num)
    print("row is correct label, column is predicted label")
    for i in range(2):
        print(" " * (11 - len(index_to_label_name[i])) + index_to_label_name[i] + ": ", end="")
        print(confusion[i] / confusion.sum())

    if train:
        print("training average IoU: ", end="")
    else:
        print("testing average IoU: ", end="")
    average_accuracy = float(np.mean(list(map(lambda x: float(x[0]), all_accuracies_images))) * 100)
    print("average balanced accuracy from all images: %3.4f%%" % average_accuracy)

    all_accuracies_images.sort(key=lambda x: x[0])

    if show_images:
        n_shown_images = 6
        shown_image_indexes = [int(i) for i in np.linspace(0, len(all_accuracies_images) - 1, n_shown_images)][::-1]

        if use_matplotlib:
            colorbar_norm = colors.Normalize(vmin=-1, vmax=1, clip=True)

            f, ax_list = plt.subplots(3, n_shown_images)
            f.set_size_inches(15,10)
            displayed_accuracies = []
            for i in range(len(ax_list[0])):
                image, label, _ = image_set[all_accuracies_images[shown_image_indexes[i]][1]]
                displayed_accuracies.append(all_accuracies_images[shown_image_indexes[i]][0])
                ax_list[0][i].imshow(image.squeeze().astype(np.uint8))

                x = model.prep_images([image])
                prediction_raw = np.tanh((model(x) if model_trt is None else model_trt(x)).data.cpu().numpy().astype(np.float32))
                im_obj = ax_list[1][i].imshow(prediction_raw.squeeze(), cmap='jet')  #, norm=colorbar_norm)
                # uncomment this line to add a color scale to the predicted plots
                # f.colorbar(im_obj, ax=ax_list[1][i])

                label[0, 0] = 1
                label[0, 1] = -1
                gt_im_obj = ax_list[2][i].imshow(label, cmap='jet', norm=colorbar_norm)
                # uncomment this line to add a colro scale to the ground truth (gt) plots
                # f.colorbar(gt_im_obj, ax=ax_list[2][i])

            plt.suptitle("Best and Worst Results from " + ("Training" if train else "Testing") + ", IoU Accuracies Are: \n" + ", ".join(map(lambda x: str(x)[:5], displayed_accuracies)))
            f.tight_layout()
            plt.savefig(training_log["output_dir"] + "output_epoch_" + str(epoch_num) + ("_train" if train else "_test") + ".png", dpi=100)
            plt.close(f)

    return average_accuracy


def get_balanced_accuracy(confusion):
    """
    confusion should be in the form
    [[true positive,  false negative]
     [false positive, true negative ]]
    """
    acc = float((((confusion[0, 0] + 0.0000001) / (confusion[0, 0] + confusion[0, 1] + 0.0000001)) +
                ((confusion[1, 1] + 0.0000001) / (confusion[1, 0] + confusion[1, 1] + 0.0000001))) / 2)
    return acc


def squared_adjusted_error(predictions, label_images, exclusion_images, train_images):
    diff = torch.abs(label_images - predictions)
    pixelwise_error = torch.max(diff - 0.5, diff * 0.01)

    # +1 where the gt image is lane and not excluded, 0 everywhere else
    positive_points = (label_images ==  1) * (exclusion_images == 0).type(torch.float32)
    # +1 where the gt image is not lane and not excluded, 0 everywhere else
    negative_points = (label_images == -1) * (exclusion_images == 0).type(torch.float32)

    pixelwise_loss = pixelwise_error * (positive_points * (1. / train_images.get_postive_negative_ratio()) + negative_points)
    loss = torch.sum(pixelwise_loss) / np.product(pixelwise_loss.size())
    return loss


def deadzone_loss(pred, label, exclusion):
    """
    
    """
    assert(len(pred.size()) == 4)
    assert(pred.size()[1] == 1)
    assert(len(label.size()) == 3)
    assert(len(exclusion.size()) == 3)
    assert(exclusion.size() == label.size())
    assert(pred.size()[:1] + pred.size()[2:] == label.size())

    pred = pred.squeeze()

    # calculate the training loss
    #TODO: explain what this loss function does
    pixelwise_error_unthresholded = torch.abs(pred - label) - 0.5
    pixelwise_loss = torch.max(pixelwise_error_unthresholded, 0.001 * pixelwise_error_unthresholded)

    # +1 where the gt image is lane and not excluded, 0 everywhere else
    positive_points = (label ==  1) * (exclusion == 0).type(torch.float32)
    # +1 where the gt image is not lane and not excluded, 0 everywhere else
    negative_points = (label == -1) * (exclusion == 0).type(torch.float32)

    # pixelwise_loss = torch.pow(predictions - label_images, 2)
    true_positives = torch.sum(positive_points * pixelwise_loss, axis=(1, 2))
    true_negatives = torch.sum(negative_points * pixelwise_loss, axis=(1, 2))

    loss_per_image = (true_positives / (torch.sum(positive_points, axis=(1, 2)) + 0.001) +
            (true_negatives / (torch.sum(negative_points, axis=(1, 2)) + 0.001))) / 2  # 0.001 added to avoid division by 0

    loss = torch.mean(loss_per_image)
    return loss


def leaky_sigmoid(x):
    linearity_factor = balanced_loss_linearity_factor
    return (1 - 2 * linearity_factor) * torch.sigmoid((balanced_loss_horizontal_scale * x).type(torch.float32)) + linearity_factor * (x + 1)
    # return torch.sigmoid(10 * x)


@memoized
def memoized_zeros(size, dtype):
    """
    This is actually pretty dangerous because the returned object is mutable. This is an optimization so that the IoU
    loss function doesn't have to create a new array of all zeros every iteration (I haven't tested how much time it
    actually saves yet)
    """
    return torch.zeros(size, dtype=dtype).to(get_default_torch_device())


@memoized
def memoized_ones(size, dtype):
    return torch.ones(size, dtype=dtype).to(get_default_torch_device())


def IoU(pred, label, exclusion, over_1_penalty=0, buffer_pixels=2):
    """
    Intersecion over Union

    pred should have shape [n, 1, x, y]
    lable and exclusion should have shape [n, x, y]

    buffer_pixels creates a margin between gt + and - pixels where no loss function is applied
    """

    try:
        assert(len(pred.size()) == 4)
        assert(pred.size()[1] == 1)
        assert(len(label.size()) == 3)
        assert(len(exclusion.size()) == 3)
        assert(exclusion.size() == label.size())
        assert(pred.size()[:1] + pred.size()[2:] == label.size())
    except AssertionError as e:
        pdb.set_trace()

    orig_shape = pred.size()

    n = orig_shape[0]

    # pred = pred.squeeze()
    if len(pred.size()) == 3:
        pass
    elif len(pred.size()) == 4:
        pred = pred[:, 0, :, :]
    else:
        assert(False)


    # +1 where the gt image is lane and not excluded, 0 everywhere else
    gt_plus = ((label ==  1) * (exclusion == 0)).type(torch.float32)
    # +1 where the gt image is not lane and not excluded, 0 everywhere else
    gt_neg = ((label == -1) * (exclusion == 0)).type(torch.float32)

    ones = memoized_ones(gt_neg.size(), torch.float)
    zeros = memoized_zeros(gt_neg.size(), torch.float)


    def elementwise_min(t1, t2):
        combined = torch.cat((t1.unsqueeze(2), t2.unsqueeze(2)), dim=2)
        return torch.min(combined, dim=2)[0].squeeze(2)


    # do the dilation
    conv = gt_plus.clone().detach()
    conv = conv.reshape((conv.shape[0], 1, conv.shape[1], conv.shape[2]))
    for i in range(buffer_pixels):
        conv = torch.nn.functional.conv2d(conv, torch.as_tensor(np.ones((1, 1, 3, 3))).type(torch.float32).to(get_default_torch_device()), stride=1, padding=1)
        conv = elementwise_min(conv, ones.reshape(conv.shape))

    extra_exclusion_map = 1 - (conv.reshape(gt_plus.shape) - gt_plus)  # 0 in the excluded border pixels, 1 everywhere else

    gt_plus *= extra_exclusion_map
    gt_neg *= extra_exclusion_map


    # true_pos = torch.min(gt_plus * f(pred * gt_plus), ones)
    # false_neg = torch.min(gt_plus * (gt_plus - f(pred * gt_plus)), ones)
    # true_neg = torch.min(gt_neg * torch.sigmoid(-1 * pred * gt_neg), ones)
    # false_pos = torch.min(gt_neg * (gt_neg - f(-1 * pred * gt_neg)), ones)

    # positive_sum = torch.sum(gt_plus, axis=(1, 2))
    # negative_sum = torch.sum(gt_neg, axis=(1, 2))

    eps = 0.00001
    image_losses = torch.zeros([pred.size()[0]], dtype=torch.float32).to(get_default_torch_device())
    for i in range(pred.size()[0]):
        if torch.sum(gt_plus) != 0:
            # intersection = torch.min(f(pred[i]), gt_plus[i])
            # union = torch.max(f(pred[i]), gt_plus[i])
            intersection = torch.sigmoid(pred[i].type(torch.float32)) * gt_plus[i]
            union = torch.sigmoid(pred[i].type(torch.float32)) + gt_plus[i] - intersection
            image_losses[i] = torch.sum(torch.min(intersection, ones)) / torch.sum(torch.max(union, zeros))
        else:
            image_losses[i] = 1 + 0.1 * (100 / (torch.sum(torch.sigmoid(pred[i].type(torch.float32)), axis=(0, 1)) + 100) - 1)

    # image_losses -= torch.sum(torch.abs(pred - label)) * 10
    return image_losses




def approx_balanced_accuracy(pred, label, exclusion, over_1_penalty=0):
    """
    Calculates an approximate but differentiable balanced accuracy (this makes it useable as a neural network loss).

    However, any individual predicted pixels outside [-1,1] will be penalized by the over_1_penalty term (so this
    function no longer approximates balanced accuracy in this case)

    Note: a quirk of balanced accuracy is that if all data is from one class then the maximum accuracy is 50%, not 100%
    (even if all data is correctly predicted)

    pred should have shape [n, 1, x, y]
    lable and exclusion should have shape [n, x, y]
    """

    try:
        assert(len(pred.size()) == 4)
        assert(pred.size()[1] == 1)
        assert(len(label.size()) == 3)
        assert(len(exclusion.size()) == 3)
        assert(exclusion.size() == label.size())
        assert(pred.size()[:1] + pred.size()[2:] == label.size())
    except AssertionError as e:
        pdb.set_trace()

    pred = pred.squeeze()
    if len(pred.size()) != 3:
        pred = torch.unsqueeze(pred, 0)

    # +1 where the gt image is lane and not excluded, 0 everywhere else
    gt_plus = (label == 1) * (exclusion == 0).type(torch.float32)
    # +1 where the gt image is not lane and not excluded, 0 everywhere else
    gt_neg = (label == -1) * (exclusion == 0).type(torch.float32)

    true_pos = torch.min(gt_plus * leaky_sigmoid(pred * gt_plus), torch.ones(gt_plus.size(), dtype=torch.float))
    false_neg = gt_plus * (gt_plus - leaky_sigmoid(pred * gt_plus))
    true_neg = torch.min(gt_neg * leaky_sigmoid(-1 * pred * gt_neg), torch.ones(gt_neg.size(), dtype=torch.float))
    false_pos = gt_neg * (gt_neg - leaky_sigmoid(-1 * pred * gt_neg))

    positive_sum = torch.sum(gt_plus, axis=(1, 2))
    negative_sum = torch.sum(gt_neg, axis=(1, 2))

    eps = 0.00001

    balanced_accuracy_each_image = ((torch.sum(true_pos, axis=(1, 2)) + eps) / (positive_sum + eps) +
            ((torch.sum(true_neg, axis=(1, 2)) + eps) / (negative_sum + eps))) / 2  # 0.00001 added to avoid division by 0

    # balanced_accuracy = torch.mean(balanced_accuracy_each_image)

    zeros_like = torch.zeros(pred.size(), dtype=torch.float).to(get_default_torch_device())
    overages = torch.abs(pred) - 1.0
    overage_penalty = torch.sum(torch.max(zeros_like, overages), axis=(1, 2)) * (over_1_penalty / np.product(pred.size()))

    augmented_accuracy = balanced_accuracy_each_image + overage_penalty

    return augmented_accuracy


def get_confusion_mat_single_image(prediction_raw, label):
    confusion = np.zeros((2, 2), dtype=np.float64)

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


def get_IoU_single_image(prediction_raw, label):

    prediction = (prediction_raw > 0)
    label_thresholded = (label > 0)

    true_positives = np.count_nonzero(label_thresholded * prediction)
    true_negatives = np.count_nonzero(~label_thresholded * ~prediction)
    false_positives = np.count_nonzero(~label_thresholded * prediction)
    false_negatives = np.count_nonzero(label_thresholded * ~prediction)

    if np.count_nonzero(label_thresholded) < 5:
        # pdb.set_trace()
        pass

    if np.count_nonzero(label_thresholded) == 0:
        # no positive pixels in the image, return normal accuracy
        return float(true_negatives) / (true_negatives + false_positives)
    else:
        return float(true_positives) / (true_positives + false_positives + false_negatives)


if __name__ == "__main__":
    main()
    # cProfile.run('main()', 'profile_results')
