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
        self.run_again = True

    def fit(self, X, N, num_epochs=200, lr=0.001, beta_o=0.1):
        self.A_ = tf.convert_to_tensor(X.T, np.float32)
        self.E_ = np.zeros((N, N))

        if self.optimizer is None:
            self.optimizer_ = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            self.optimizer_ = self.optimizer

        self.losses_ = {
            "min_in": [],
            "max_out": [],
            "d_min": [],
            # "first": [],
            "d_max": [],
            "out": [],
            "E": [],
            # "all": [],
        }

        self.loss_value_ = np.inf
        self.run_again = False
        pbar = tqdm(range(num_epochs+1), disable=self.verbose)
        for epoch in pbar: #range(num_epochs + 1):
            loss_value, grads, Ei, run_again = self._grad(epoch, beta_o)
            self.optimizer_.apply_gradients(zip(grads, self.kmodel.trainable_variables))
            pbar.set_description(f"Epoch: {epoch} - Loss: {loss_value:.2f}")
            if loss_value < self.loss_value_:
                self.Ei_ = Ei
                self.loss_value_ = loss_value
            if run_again:
                self.run_again = True
                return self
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
                if self.Ei_[i, j] > 0:
                    we.append((i, j, self.Ei_[i, j]))
        G.add_weighted_edges_from(we)
        w = []
        pos_final = {}
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

        plt.figure(figsize=[5, 4])
        epochs = np.arange(0, len(self.losses_["d_min"]))
        cmap = sns.color_palette(sns.color_palette("hls", len(self.losses_)))
        for i, (label, ls) in enumerate(self.losses_.items()):
            plt.plot(epochs, ls, label=label)#, hue=i, palette=cmap, hue_order=self.losses_.keys(), alpha=0.8)
        plt.title("Losses")
        plt.legend()
        plt.savefig("losses.png")
        plt.show()
        plt.clf()
        plt.close()
        gc.collect()


    def _loss(self, output, epoch, beta_o):
        A = tf.convert_to_tensor(tf.transpose(self.A_), np.float32)
        D = _squared_dist(A, tf.transpose(output))
        d_min = tf.math.reduce_min(D, axis=1)
        # d_max = tf.math.reduce_max(D, axis=1)
        N = self.E_.shape[0]
        Et = np.zeros((N, N))

        s = tf.argsort(D.numpy(), axis=1)[:, :2].numpy()
        extra_cost = False
        alpha = 1
        min_inside = tf.Variable(tf.zeros((N,), dtype=np.float32))
        max_outside = tf.Variable(tf.zeros((N,), dtype=np.float32))
        d_max = tf.Variable(tf.zeros((N,), dtype=np.float32))
        for i in range(N):
            idx = s[:, 0] == i
            si = s[idx]
            if len(si) == 0:
                extra_cost = False
            else:
                a = A[idx]
                b = A[~idx]
                d_max[i].assign(tf.math.reduce_max(_squared_dist(a, tf.expand_dims(output[:, i], axis=0))))
                min_inside[i].assign(tf.reduce_max(_squared_dist(a, a)))
                max_outside[i].assign(tf.reduce_min(_squared_dist(a, b)))
                for j in set(si[:, 1]):
                    k = sum(si[:, 1] == j)
                    Et[i, j] += k  # alpha * (k - Et[i, j])
                    Et[j, i] += k  # alpha * (k - Et[j, i])

        O = _squared_dist(output, output) + 0.01
        o_min = tf.math.reduce_min(O, axis=1) + 0.1

        E = tf.convert_to_tensor(Et, np.float32)
        # min_inside = tf.convert_to_tensor(np.array(min_inside), np.float32)
        # max_outside = tf.convert_to_tensor(np.array(max_outside), np.float32)

        if epoch == 0:
            self.min_in_0_ = 1 * tf.reduce_max(min_inside)
            self.max_out_0_ = 1 * tf.reduce_max(max_outside)
            # self.d_min_0_ = 1 * tf.norm(d_min, 2)
            # self.d_max_0_ = 1 * tf.norm(d_max, 2)
            self.d_min_0_ = 1 * tf.norm(d_min)
            self.d_max_0_ = 1 * tf.norm(d_max)
            self.out_0_ = tf.norm(O, 2)
            self.El_0_ = 1 * tf.norm(E, 1)

        min_in = 1 * tf.reduce_max(min_inside) / self.min_in_0_
        max_out = 1 * tf.reduce_max(max_outside) / self.max_out_0_
        # d_min = 1 * tf.norm(d_min, 2) / self.d_min_0_ # più è piccola più il vincente è vicino
        # d_max = 1 * tf.norm(d_max, 2) / self.d_max_0_
        d_min = 1 * tf.norm(d_min) / self.d_min_0_ # più è piccola più il vincente è vicino
        d_max = 1 * tf.norm(d_max) / self.d_max_0_
        # out = tf.norm(O, 2) / self.out_0_ # più è piccola più i neuroni sono lontani
        # El = 1 * tf.norm(tf.cast(tf.not_equal(E, 0), dtype='float32')) / self.El_0_
        El = 1 * tf.norm(E, 1) / self.El_0_
        # 0.01 * tf.norm(D) / self.losses_["d_max"]
        # cost = min_in / max_out * d_min + d_max + out + El
        # cost = min_in / max_out + d_min + El
        cost = min_in / max_out + d_min + d_max + El

        self.losses_["min_in"].append(min_in.numpy())
        self.losses_["max_out"].append(max_out.numpy())
        self.losses_["d_min"].append(d_min.numpy())
        self.losses_["d_max"].append(d_max.numpy())
        # self.losses_["out"].append(out.numpy())
        self.losses_["E"].append(El.numpy())
        # self.losses_["first"].append(0.4 * tf.norm(d_min, 2).numpy())
        # self.losses_["all"].append(cost.numpy())

        return cost, Et, extra_cost

    def _grad(self, epoch, beta_o):
        with tf.GradientTape() as tape:
            loss_value, Ei, run_again = self._loss(self.kmodel(self.A_), epoch, beta_o)
        return loss_value, tape.gradient(loss_value, self.kmodel.trainable_variables), Ei, run_again


def _squared_dist(A, B):
    row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
    row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

    row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
    row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

    return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B
