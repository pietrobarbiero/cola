import gc

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


class Fexin():
    def __init__(self, kmodel, optimizer=None, verbose=True):
        self.kmodel = kmodel
        self.optimizer = optimizer
        self.verbose = verbose

    def fit(self, X, N, num_epochs=200, lr=0.001, beta_o=0.1):
        self.A_ = tf.convert_to_tensor(X.T, np.float32)
        self.E_ = np.zeros((N, N))

        if self.optimizer is None:
            self.optimizer_ = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            self.optimizer_ = self.optimizer

        self.loss_value_ = np.inf
        # pbar = tqdm(range(num_epochs+1), disable=self.verbose)
        for epoch in range(num_epochs + 1):
            loss_value, grads, Ei = _grad(self.kmodel, self.A_, self.E_, epoch, beta_o)
            self.optimizer_.apply_gradients(zip(grads, self.kmodel.trainable_variables))
            # if self.verbose:
            #     pbar.set_description(f"Epoch: {epoch} - Loss: {loss_value:.2f}")
            if loss_value < self.loss_value_:
                self.Ei_ = Ei
                self.loss_value_ = loss_value
        return self

    def predict(self, X):
        return self.kmodel(X).numpy().T

    def plot(self, X, y=None, title="", file_path="fexin.png"):
        Wa = self.predict(self.A_)

        D = euclidean_distances(X, Wa)
        N = self.E_.shape[0]
        self.Ei_ = np.zeros((N, N))
        s = np.argsort(D, axis=1)[:, :2]
        for i in range(len(self.Ei_)):
            si = s[s[:, 0] == i]
            if len(si) > 0:
                for j in set(si[:, 1]):
                    k = sum(si[:, 1] == j)
                    self.Ei_[i, j] += k  # alpha * (k - Et[i, j])
                    self.Ei_[j, i] += k  # alpha * (k - Et[j, i])

        if X.shape[1] > 2:
            tsne = TSNE(n_components=2)
            M_list = [X]
            for i in range(Wa.shape[0]):
                M_list.append(Wa[i].reshape(1, -1))
            M = np.concatenate(M_list)
            Mp = tsne.fit_transform(M)
            Xp = Mp[:X.shape[0]]
            Wp = Mp[X.shape[0]:]
            pos = {}
            for i in range(len(Wp)):
                pos[i] = Wp[i].reshape(1, -1)[0]
        else:
            Xp = X
            pos = {}
            for i in range(Wa.shape[0]):
                pos[i] = Wa[i]

        G = nx.Graph()
        we = []
        for i in range(0, self.Ei_.shape[0]):
            for j in range(i+1, self.Ei_.shape[1]):
                we.append((i, j, self.Ei_[i, j]))
        G.add_weighted_edges_from(we)
        w = []
        for e in G.edges:
            w.append(self.Ei_[e[0], e[1]])
        wd = np.array(w)
        widths = MinMaxScaler(feature_range=(0, 5)).fit_transform(wd.reshape(-1, 1)).squeeze().tolist()

        plt.figure(figsize=[5, 4])
        if y is not None:
            cmap = sns.color_palette(sns.color_palette("hls", len(set(y))))
            sns.scatterplot(Xp[:, 0], Xp[:, 1], hue=y, palette=cmap, hue_order=set(y), alpha=0.8)
        else:
            sns.scatterplot(Xp[:, 0], Xp[:, 1])
        c = '#00838F'
        nx.draw_networkx_nodes(G, pos=pos, node_size=600, node_color=c)
        nx.draw_networkx(G, pos=pos, node_size=0, width=0, font_color='white', font_weight="bold")
        nx.draw_networkx_edges(G, pos=pos, width=widths, edge_color=c)
        plt.title(title)
        plt.savefig(file_path)
        plt.show()
        plt.clf()
        plt.close()
        gc.collect()


def _squared_dist(A, B):
    row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
    row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

    row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
    row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

    return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B


def _loss(X, output, Et, epoch, beta_o):
    A = tf.convert_to_tensor(tf.transpose(X), np.float32)
    D = _squared_dist(A, tf.transpose(output))
    d_min = tf.math.reduce_min(D, axis=1)
    d_max = tf.math.reduce_max(D, axis=1)
    N = Et.shape[0]
    Et = np.zeros((N, N))

    s = tf.argsort(D.numpy(), axis=1)[:, :2].numpy()
    extra_cost = 1
    alpha = 1
    for i in range(len(Et)):
        si = s[s[:, 0] == i]
        if len(si) == 0:
            extra_cost *= 2
        else:
            for j in set(si[:, 1]):
                k = sum(si[:, 1] == j)
                Et[i, j] += k # alpha * (k - Et[i, j])
                Et[j, i] += k # alpha * (k - Et[j, i])

    O = _squared_dist(output, output) + 0.01
    o_min = tf.math.reduce_min(O, axis=1) + 0.1

    E = tf.convert_to_tensor(Et, np.float32)

    return extra_cost * (
                         0.4 * tf.norm(d_min) +
                         0.001 * tf.norm(d_max) +
                         0.01 * tf.norm(D) +
                         beta_o * 1/tf.norm(O) +
                         0.4 * tf.norm(E)
                        ), \
           Et


def _grad(model, inputs, Et, epoch, beta_o):
    with tf.GradientTape() as tape:
        loss_value, Et = _loss(inputs, model(inputs), Et, epoch, beta_o)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), Et
