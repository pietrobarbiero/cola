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


def quantization_fast(input, output, y, O, epoch):
    # TODO:
    # Q = norm(X[i] - output[j]) * O[i,j]
    D = squared_dist(input, tf.transpose(output))
    D2 = tf.multiply(D, O)
    # togliere il bias
    Q = tf.norm(D2)

    # classic
    # mettere a zero tutto ciò che non è nel voronoi
    # togliere il bias
    # D = squared_dist(input, tf.transpose(output))
    # d_min = tf.math.reduce_min(D, axis=1)
    # Q = tf.norm(d_min)

    import matplotlib.pyplot as plt
    import seaborn as sns
    if (epoch % 100) == 0:
    # if True:
        # plt.figure()
        # plt.subplot(121)
        # sns.heatmap(O.numpy())
        # plt.subplot(122)
        # sns.heatmap(y.reshape(-1,1))
        # plt.title(f'Epoch: {epoch} - Bias: {tf.norm(b).numpy():.4f}')
        # plt.show()

        plt.figure(figsize=[8,3])
        for i in range(O.numpy().shape[1]):
            wi = O.numpy()[:, i]
            plt.subplot(1, O.numpy().shape[1], i+1)
            plt.scatter(input.numpy()[:, 0], input.numpy()[:, 1], c='k', alpha=0.2)
            plt.scatter(input.numpy()[wi>0, 0], input.numpy()[wi>0, 1], c='g')
            plt.scatter(input.numpy()[wi<=0, 0], input.numpy()[wi<=0, 1], c='r')
            plt.scatter(output[0, 0], output[1, 0], c='k', s=200, alpha=0.2)
            plt.scatter(output[0, 1], output[1, 1], c='k', s=200, alpha=0.2)
            plt.scatter(output[0, 2], output[1, 2], c='k', s=200, alpha=0.2)
            plt.scatter(output[0, i], output[1, i], c='k', s=200)
            plt.title(f'No Voronoi - epoch {epoch}')
            # plt.title(f'Voronoi - epoch {epoch}')
        plt.savefig(f'novoronoi_{epoch}.png')
        # plt.savefig(f'voronoi_{epoch}.png')
        plt.show()

        # plt.figure()
        # plt.scatter(input.numpy()[y==0, 0], input.numpy()[y==0, 1], c='r', alpha=0.5)
        # plt.scatter(input.numpy()[y==1, 0], input.numpy()[y==1, 1], c='g', alpha=0.5)
        # plt.scatter(input.numpy()[y==2, 0], input.numpy()[y==2, 1], c='b', alpha=0.5)
        # plt.scatter(output[0, 0], output[1, 0], c='r', s=200)
        # plt.scatter(output[0, 1], output[1, 1], c='g', s=200)
        # plt.scatter(output[0, 2], output[1, 2], c='b', s=200)
        # plt.show()

        # t1 = np.arange(sum(y==0))
        # t2 = np.arange(sum(y==1))
        # t3 = np.arange(sum(y==2))
        # plt.figure()
        # plt.plot(t1, O.numpy()[y==0, 0], c='r')
        # plt.plot(t2, O.numpy()[y==1, 0], c='g')
        # plt.plot(t3, O.numpy()[y==2, 0], c='b')
        # plt.show()
        # plt.figure()
        # plt.plot(t1, O.numpy()[y==1, 0], c='r')
        # plt.plot(t1, O.numpy()[y==1, 1], c='g')
        # plt.plot(t1, O.numpy()[y==1, 2], c='b')
        # plt.show()
    #
    #     print()

    return Q


def convex_hull_loss(input, output):
    # TODO: try again without bias
    Z = squared_dist(input, tf.transpose(output))
    Z2 = tf.divide(1, Z)
    # Z = tf.sqrt(squared_dist(input, tf.transpose(output)))
    v = tf.reduce_sum(Z2, axis=1)
    v2 = tf.reshape(tf.tile(v, [Z2.shape[1]]), [Z2.shape[1], v.shape[0]])
    v2 = tf.transpose(v2)
    Zh = tf.divide(Z2, v2)
    Ph = tf.transpose(Zh)

    # A = tf.reduce_max(P, axis=0)
    # mask = tf.logical_not(tf.less(P, A))
    # Ph = tf.multiply(P, tf.cast(mask, P.dtype))
    # Ph = tf.math.divide_no_nan(Ph, Ph)

    K = tf.matmul(output, Ph)
    Q = tf.norm(input - tf.transpose(K)) + 0.001 * tf.norm(Z)

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
