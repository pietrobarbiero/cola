import tensorflow as tf
import numpy as np

from ._utils import squared_dist


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


def convex_hull_loss(input, output):
    Z = squared_dist(input, tf.transpose(output))
    Z2 = tf.divide(1, Z)
    # Z = tf.sqrt(squared_dist(input, tf.transpose(output)))
    v = tf.reduce_sum(Z2, axis=1)
    v2 = tf.reshape(tf.tile(v, [Z2.shape[1]]), [Z2.shape[1], v.shape[0]])
    v2 = tf.transpose(v2)
    Zh = tf.divide(Z2, v2)
    P = tf.transpose(Zh)

    A = tf.reduce_max(P, axis=0)
    mask = tf.logical_not(tf.less(P, A))
    Ph = tf.multiply(P, tf.cast(mask, P.dtype))
    Ph = tf.math.divide_no_nan(Ph, Ph)

    K = tf.matmul(output, Ph)
    Q = tf.norm(input - tf.transpose(K)) # + 0.001 * tf.norm(Z)

    # import matplotlib.pyplot as plt
    # y = output.numpy().T
    # z = K.numpy()[:, 0]
    # x = input.numpy()[0, :]
    # plt.figure()
    # plt.scatter(x[0], x[1], label='x')
    # plt.scatter(y[:, 0], y[:, 1], label='y')
    # plt.scatter(z[0], z[1], label='yh')
    # plt.legend()
    # plt.show()

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
