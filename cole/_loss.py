import tensorflow as tf
import numpy as np


def qe_loss(input, output):
    # adjacency_matrix = np.zeros((self.N, self.N))
    D = _squared_dist(input, tf.transpose(output))
    d_min = tf.math.reduce_min(D, axis=1)
    # s = tf.argsort(D.numpy(), axis=1)[:, :2].numpy()
    # for i in range(self.N):
    #     idx = s[:, 0] == i
    #     si = s[idx]
    #     if len(si) > 0:
    #         for j in set(si[:, 1]):
    #             k = sum(si[:, 1] == j)
    #             adjacency_matrix[i, j] = 1
    #             adjacency_matrix[j, i] = 1
    #
    # E = tf.convert_to_tensor(adjacency_matrix, np.float32)
    Eq = tf.norm(d_min)
    # El = tf.norm(E, 2)
    # cost = Eq #+ self.lmb * El
    # self.loss_Q_.append(Eq.numpy())
    # self.loss_E_.append(El.numpy())
    return Eq


def _squared_dist(A, B):
    row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
    row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

    row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
    row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

    return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B
