import tensorflow as tf
import numpy as np

from cole._utils import squared_dist


def silhouette(input, output):
    N = input.shape[0]
    D = squared_dist(input, tf.transpose(output))
    b = tf.math.reduce_min(D, axis=1)
    Q = tf.norm(b)
    prototype_idx = tf.argmin(D, axis=1)
    si = tf.Variable(0, dtype='float32')
    for i in range(N):
        cluster_points = input[prototype_idx==prototype_idx[i]]
        ai = tf.norm(cluster_points - input[i], ord=2)
        maxi = tf.maximum(ai, b[i])
        si = tf.add(si, (b[i] - ai) / maxi)
    si = -tf.divide(si, N)
    return si


def quantization(input, output):
    D = squared_dist(input, tf.transpose(output))
    d_min = tf.math.reduce_min(D, axis=1)
    Q = tf.norm(d_min)
    return Q


def quantization_topology(input, output, lmb):
    N = input.shape[0]
    adjacency_matrix = np.zeros((N, N))
    D = squared_dist(input, tf.transpose(output))
    d_min = tf.math.reduce_min(D, axis=1)
    s = tf.argsort(D.numpy(), axis=1)[:, :2].numpy()
    for i in range(N):
        idx = s[:, 0] == i
        si = s[idx]
        if len(si) > 0:
            for j in set(si[:, 1]):
                k = sum(si[:, 1] == j)
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    E = tf.convert_to_tensor(adjacency_matrix, np.float32)
    Q = tf.norm(d_min)
    E = tf.norm(E, 2)
    cost = Q + lmb * E
    return cost
