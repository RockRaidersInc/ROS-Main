import numpy as np
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from functools import reduce
import warnings

from confusion_mat_tools import save_confusion_matrix


def read_dataset(filename):
    # The dataset features are stored in a .npz file, np.load will give us a dictionary-like object with all the features
    train_set_raw = np.load(filename)
    features = []
    labels = []
    file_names = []

    for key in train_set_raw:
        features.append(train_set_raw[key])
        labels.append(key.split("/")[2])
        file_names.append(key)

    return labels, features, file_names


def get_single_var_confusion(train_labels, train_features, test_labels, test_features, target_class, c_vals, test_file_names=None):
    """
    Trains a SVM for each passed value of C (in c_vals), finds the best one, and returns a single row of the
    confusion matrix (for target_class).
    """

    # first find the best value of C. Do this by splitting the training data up into a training set and validation set
    print("now finding the vest value of C for class " + target_class)
    def train_and_test(C):
        # divide the train set up into a new training and validation set (for testing different values of C)
        n_train_features = int(len(train_features) * 0.8)
        new_train_features = train_features[:n_train_features]
        new_train_labels = train_labels[:n_train_features]
        new_validation_features = train_features[n_train_features:]
        new_validation_labels = train_labels[n_train_features:]

        # now train a SVM with the new train and validation sets
        clf = train_svm(target_class, new_train_features, new_train_labels, C)
        correct_ratio, predictions = test_svm(clf, target_class, new_validation_features, new_validation_labels)
        print("\twith C=%1.1f: labeled %3.5f%% of validation set accurately" % (C, correct_ratio * 100))
        return correct_ratio, C

    # train an evaluate an SVM for every value of c (the list defined at the start of the file)
    all_svm_results = list(map(train_and_test, c_vals))
    _, best_c = max(all_svm_results, key=lambda x: x[0])


    # now train a new SVM on all the training data with the best value of C
    clf = train_svm(target_class, train_features, train_labels, best_c)
    best_accuracy, best_predictions = test_svm(clf, target_class, test_features, test_labels)

    # print out the best results:
    print("best SVM: got %3.1f%% of predictions correct on test data with C=%1.1f" % (best_accuracy * 100, best_c))

    # optionally print out the file names of a false positive and a false negative
    if test_file_names is not None:
        bool_test_labels = np.array([1 if i == target_class else -1 for i in test_labels])
        false_positives = (best_predictions != bool_test_labels) * (bool_test_labels == -1)
        false_negatives = (best_predictions != bool_test_labels) * (bool_test_labels == 1)
        print("false positive:", test_file_names[np.nonzero(false_positives)[0][0]])
        print("false negative:", test_file_names[np.nonzero(false_negatives)[0][0]])

    print()  # a newline to separate classes

    # put together a row of the confusion matrix
    confusion_dict = {}
    for i in range(len(test_labels)):
        if best_predictions[i] == 1:  # if the SVM predicted the right value
            if test_labels[i] in confusion_dict:
                confusion_dict[test_labels[i]] += 1
            else:
                confusion_dict[test_labels[i]] = 1

    return confusion_dict


def test_svm(clf, target_class, test_features, test_labels):
    """
    Evaluates the passed SVM on the passed data. The proportion of test vectors it correctly labeled and
    a matrix of predicted labels are returned.
    """
    test_predictions = clf.predict(test_features)

    # will be 1 where test_predictions and processed_test_labels are the same and 0 everywhere else
    test_labels = np.array([1 if i == target_class else -1 for i in test_labels])
    correct_predictions = test_predictions == test_labels
    correct_ratio = correct_predictions.mean()
    return correct_ratio, test_predictions


def train_svm(target_class, train_features, train_labels, c):
    """
    Trains and returns an SVM on the passed data and passed value of c.
    """
    processed_train_labels = np.array([1 if i == target_class else -1 for i in train_labels])
    clf = LinearSVC(C=c, random_state=1)  # random_state=1 is important, this keeps results from varrying across runs
    clf.fit(train_features, processed_train_labels)
    return clf


def main(c_vals=(1,)):
    global test_file_names

    # load the datasets
    train_labels, train_features, _ = read_dataset("image_descriptors_train.npz")
    test_labels, test_features, test_file_names = read_dataset("image_descriptors_test.npz")

    # make a python set with all the class names ("redcarpet", "grass", ect)
    all_class_labels = list(reduce(lambda a, b: a | {b}, test_labels, set()))

    confusion_dicts = {}

    for label in all_class_labels:
        confusion_dicts[label] = get_single_var_confusion(train_labels,
                                                          train_features,
                                                          test_labels,
                                                          test_features,
                                                          label,
                                                          c_vals,
                                                          test_file_names=test_file_names)

    # now print the confusion matrix (and turn it into a numpy matrix while we're at it)
    confusion_matrix_list = []
    # first print the x axis lables (what the network predicted
    print("Confusion Matrix:")
    print(" " * 11, end="")
    for y in all_class_labels:
        print(" " * (11 - len(y)) + y, end="")
    print()
    # now print the y lables and values (the actual class labels)
    for y in all_class_labels:
        predictions = confusion_dicts[y]
        print(" " * (11 - len(y)) + y, end="")
        for x in all_class_labels:
            print("%11i" % (predictions[x],), end="")
            confusion_matrix_list.append(predictions[x])
        print()

    # now turn the confusion matrix into a numpy matrix
    confusion_mat = np.array(confusion_matrix_list).reshape(len(all_class_labels), len(all_class_labels))
    save_confusion_matrix(confusion_mat, all_class_labels, "SVM_confusion.JPEG")


if __name__ == "__main__":
    # turn off convergence warnings (they make it really hard to see actual output)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # this will create an array of 11 values from 0.1 to 10. (with approximately as many points in the
    # range [0.1, 1] and [1, 10]. These are the values of c that will be tried when tuning the SVMs
    c_vals = [10 ** float(x) for x in np.linspace(-1, 1, 11)]

    main(c_vals=c_vals)