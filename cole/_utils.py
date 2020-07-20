import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import numpy as np


def compute_confusion_matrix(X, P, y):
    N = X.shape[0]
    input_matrix = tf.Variable(X, dtype='float32')
    D = squared_dist(input_matrix, tf.transpose(P))
    s = tf.argmin(D.numpy(), axis=1).numpy()
    sz = len(set(y))
    confusion_matrix = np.zeros((sz, sz))
    for i in range(N):
        idx = s == i
        if sum(idx) > 0:
            counts = collections.Counter(y[idx])
            km, vm = counts.most_common(1)[0]
            for k, v in counts.items():
                confusion_matrix[km, k] += v
    return confusion_matrix


def plot_confusion_matrix(X, P, y, title='', file_name=None, figsize=[5, 5]):
    confmat = compute_confusion_matrix(X, P, y)
    accuracy = score(X, P, y)
    title = f'Accuracy: {accuracy:.4f}'
    plt.figure(figsize=figsize)
    sns.heatmap(confmat.astype('int'), annot=True, fmt='d',
                cbar=False, square=True, cmap='Greens')
    plt.title(title)
    plt.ylabel('true')
    plt.xlabel('predicted')
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()
    return


def score(X, P, y):
    confusion_matrix = compute_confusion_matrix(X, P, y)
    accuracy = sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    return accuracy


def squared_dist(A, B):
    row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
    row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

    row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
    row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

    return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B
