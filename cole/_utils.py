import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import numpy as np
from sklearn.manifold import TSNE


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


def plot_confusion_matrix(X, P, y):
    confmat = compute_confusion_matrix(X, P, y)
    accuracy = score(X, P, y)
    title = f'Accuracy: {accuracy:.4f}'
    sns.heatmap(confmat.astype('int'), annot=True, fmt='d',
                cbar=False, square=True, cmap='Greens')
    plt.title(title)
    plt.ylabel('true')
    plt.xlabel('predicted')
    plt.tight_layout()


def score(X, P, y):
    confusion_matrix = compute_confusion_matrix(X, P, y)
    accuracy = sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    return accuracy


def compute_graph(X, P, return_has_sampels=False):
    has_samples = []
    # N = X.shape[0]
    N = P.shape[1]
    adjacency_matrix = np.zeros((N, N))
    input_matrix = tf.Variable(X, dtype='float32')
    D = squared_dist(input_matrix, tf.transpose(P))
    s = tf.argsort(D.numpy(), axis=1)[:, :2].numpy()
    for i in range(N):
        idx = s[:, 0] == i
        if sum(idx) > 0:
            has_samples.append(True)
        else:
            has_samples.append(False)
        si = s[idx]
        if len(si) > 0:
            for j in set(si[:, 1]):
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    G = nx.Graph()
    we = []
    for i in range(0, adjacency_matrix.shape[0]):
        for j in range(i + 1, adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] > 0 and has_samples[i] and has_samples[j]:
                we.append((i, j, adjacency_matrix[i, j]))
    G.add_weighted_edges_from(we)
    if return_has_sampels:
        return G, has_samples
    else:
        return G


def scatterplot(X, prototypes, y, valid=True, links=True):
    G = compute_graph(X, prototypes)

    if X.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42)
        M_list = [X]
        nodes_idx = []
        nodes_number = []
        for i, node in enumerate(G.nodes):
            nodes_idx.append(i)
            nodes_number.append(node)
            M_list.append(prototypes[:, node].reshape(1, -1))
        M = np.concatenate(M_list)
        Mp = tsne.fit_transform(M)
        Xp = Mp[:X.shape[0]]
        Wp = Mp[X.shape[0]:]
        pos = {}
        for i in range(len(Wp)):
            pos[nodes_number[i]] = Wp[nodes_idx[i]].reshape(1, -1)[0]
    else:
        Xp = X
        Wp = prototypes.T
        pos = {}
        for i in G.nodes:
            pos[i] = prototypes[:, i]

    if valid:
        Wp = Wp[G.nodes]

    c = '#00838F'
    fig, ax = plt.subplots()
    cmap = sns.color_palette(sns.color_palette("hls", len(set(y))))
    sns.scatterplot(Xp[:, 0], Xp[:, 1], hue=y, palette=cmap, hue_order=set(y), alpha=0.3, legend=False)
    plt.scatter(Wp[:, 0], Wp[:, 1], s=200, c=c)
    if links:
        nx.draw_networkx_edges(G, pos=pos, width=2, edge_color=c)
    ax.axis('off')
    plt.tight_layout()


def squared_dist(A, B):
    row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
    row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

    row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
    row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

    return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B
