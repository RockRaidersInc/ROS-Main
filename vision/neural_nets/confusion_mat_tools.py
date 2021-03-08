# import  matplotlib.pyplot as plt
# import itertools
# import numpy as np
#
#
# def save_confusion_matrix(confusion_mat, class_names, filename):
#     """
#     Saves an image of a confusion matrix. This code is very heavily based on
#     https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
#     :param confusion_mat: A numpy confusion matrix. Should be square
#     :param class_names: A list of class names. Should have the same length as the passed confusion matrix (one axis)
#     :param filename: The file name to save the confusion matrix under (should have an image extension)
#     """
#     plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title("Confusion matrix for neural net classifier")
#     tick_marks = np.arange(len(class_names))
#     plt.xticks(tick_marks, class_names, rotation=45)
#     plt.yticks(tick_marks, class_names)
#
#     thresh = confusion_mat.max() / 2.
#     for i, j in itertools.product(range(confusion_mat.shape[0]), range(confusion_mat.shape[1])):
#         plt.text(j, i, format(confusion_mat[i, j], 'd'),
#                  horizontalalignment="center",
#                  color="white" if confusion_mat[i, j] > thresh else "black")
#
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()
#     plt.show()
#     # plt.savefig(filename)