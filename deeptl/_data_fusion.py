from keras.utils import plot_model
from sklearn.manifold import TSNE
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import collections


class Fexin():
    def __init__(self, optimizer=None, verbose=True):
        self.optimizer = optimizer
        self.verbose = verbose

    def fit(self, X_list, N_min=None, N_max=None, num_epochs=200, lr=0.01, beta_o=0.1):
        n = X_list[0].shape[0]

        self.best_loss_ = np.inf
        self.gexin_ = None
        if self.optimizer is None:
            self.optimizer_ = tf.keras.optimizers.Adagrad(learning_rate=lr)
        else:
            self.optimizer_ = self.optimizer

        if N_min is None:
            N_min = 2

        if N_max is None:
            N_max = n

        # pbar = tqdm(range(N_min, N_max))
        for level in range(N_min, N_max): #pbar:
            self.E_ = np.zeros((level, level))
            self.A_list_ = []
            xo_list = []
            input_list = []
            output_list = []
            for X in X_list:
                input = tf.keras.layers.Input(shape=(X.shape[0],))
                x = tf.keras.layers.BatchNormalization()(input)
                x = tf.keras.layers.Dense(10, activation='relu')(x)
                x = tf.keras.layers.Dense(10, activation='relu')(x)
                output = tf.keras.layers.Dense(level)(x)
                input_list.append(input)
                xo_list.append(x)
                output_list.append(output)
                self.A_list_.append(tf.convert_to_tensor(X.T, np.float32))
            self.kmodel = tf.keras.Model(inputs=input_list, outputs=output_list)
            tf.keras.utils.plot_model(self.kmodel, to_file='model.png', show_shapes=True)

            pbar = tqdm(range(num_epochs + 1))
            for epoch in pbar:  # range(num_epochs + 1):
                loss_value, grads, Ei_list = self._grad(epoch, beta_o)
                self.optimizer_.apply_gradients(zip(grads, self.kmodel.trainable_variables))
                pbar.set_description(f"Epoch: {epoch} - Loss: {loss_value:.2f}")
                if loss_value < self.best_loss_:
                    self.Ei_list_ = Ei_list
                    self.best_loss_ = loss_value

        return self

    def predict(self, X_list):
        return self.kmodel(X_list)

    def _grad(self, epoch, beta_o):
        with tf.GradientTape() as tape:
            loss_value, Ei_list = self._loss(self.kmodel(self.A_list_), epoch, beta_o)
        return loss_value, tape.gradient(loss_value, self.kmodel.trainable_variables), Ei_list

    def _loss(self, output_list, epoch, beta_o):
        N = self.E_.shape[0]
        cost = tf.Variable(0, dtype=np.float32)
        Et = np.zeros((N, N))
        for q, (A, output) in enumerate(zip(self.A_list_, output_list)):
            # Et = np.zeros((N, N))
            A = tf.transpose(A)
            D = _squared_dist(A, tf.transpose(output))
            d_min = tf.math.reduce_min(D, axis=1)

            s = tf.argsort(D.numpy(), axis=1)[:, :2].numpy()
            min_inside = tf.Variable(tf.zeros((N,), dtype=np.float32))
            max_outside = tf.Variable(tf.zeros((N,), dtype=np.float32))
            d_max = tf.Variable(tf.zeros((N,), dtype=np.float32))
            for i in range(N):
                idx = s[:, 0] == i
                si = s[idx]
                if len(si) > 0:
                    a = A[idx]
                    b = A[~idx]
                    d_max[i].assign(tf.math.reduce_max(_squared_dist(a, tf.expand_dims(output[:, i], axis=0))))
                    min_inside[i].assign(tf.reduce_max(_squared_dist(a, a)))
                    max_outside[i].assign(tf.reduce_min(_squared_dist(a, b)))
                    for j in set(si[:, 1]):
                        k = sum(si[:, 1] == j)
                        Et[i, j] += k
                        Et[j, i] += k

            # Et_list.append(Et)
            E = tf.convert_to_tensor(Et, np.float32)

            Fn = tf.reduce_max(min_inside)
            Fd = tf.reduce_max(max_outside)
            Eq = tf.norm(d_min)
            Eq2 = tf.norm(d_max)
            El = tf.norm(E, 1)
            cost = tf.add(Fn / Fd + Eq + Eq2 + El, cost)
        return cost, Et

    def plot(self, X_list, y=None, title="", file_path="fexin.png"):
        Wa_list = self.predict(self.A_list_)
        N = self.E_.shape[0]
        cmap = sns.color_palette(sns.color_palette("hls", len(set(y))))
        n = X_list[0].shape[0]
        A_samples = np.zeros((n, n))

        for k, (A, X, Wa) in enumerate(zip(self.A_list_, X_list, Wa_list)):
            Ei = self.Ei_list_
            node_colors = {}
            for i in range(N):
                node_colors[i] = []
            has_samples = []
            A = tf.transpose(A)
            D = _squared_dist(A, tf.transpose(Wa))
            s = tf.argsort(D.numpy(), axis=1)[:, :2].numpy()
            for i in range(N):
                idx = s[:, 0] == i
                if sum(idx) > 0:
                    has_samples.append(True)
                    yi = y[idx]
                    most_common_label = collections.Counter(yi).most_common(1)[0][0]
                    node_colors[i] = cmap[most_common_label]
                else:
                    has_samples.append(False)

            G = nx.Graph()
            we = []
            for i in range(0, Ei.shape[0]):
                for j in range(i+1, Ei.shape[1]):
                    if Ei[i, j] > 0 and has_samples[i] and has_samples[j]:
                        we.append((i, j, Ei[i, j]))
            G.add_weighted_edges_from(we)
            w = []
            for e in G.edges:
                w.append(Ei[e[0], e[1]])
            wd = np.array(w)
            widths = MinMaxScaler(feature_range=(0, 5)).fit_transform(wd.reshape(-1, 1)).squeeze().tolist()

            node_colors_list = []
            for node in G.nodes:
                node_colors_list.append(node_colors[node])

            if self.verbose:
                pos = nx.drawing.layout.spring_layout(G)
                plt.figure(figsize=[5, 4])
                c = '#00838F'
                nx.draw_networkx_nodes(G, pos=pos, node_size=600, node_color=node_colors_list)
                nx.draw_networkx(G, pos=pos, node_size=0, width=0, font_color='white', font_weight="bold")
                nx.draw_networkx_edges(G, pos=pos, width=widths, edge_color=c)
                plt.title(title)
                plt.savefig(f"{file_path[:-3]}_G_{k}.png")
                plt.show()
                plt.clf()
                plt.close()
                gc.collect()

                if X.shape[1] > 2:
                    tsne = TSNE(n_components=2, random_state=42)
                    M_list = [X]
                    nodes_idx = []
                    nodes_number = []
                    for i, node in enumerate(G.nodes):
                        nodes_idx.append(i)
                        nodes_number.append(node)
                        M_list.append(Wa[:, node].numpy().reshape(1, -1))
                    M = np.concatenate(M_list)
                    Mp = tsne.fit_transform(M)
                    Xp = Mp[:X.shape[0]]
                    Wp = Mp[X.shape[0]:]
                    pos = {}
                    for i in range(len(Wp)):
                        pos[nodes_number[i]] = Wp[nodes_idx[i]].reshape(1, -1)[0]
                else:
                    Xp = X
                    pos = {}
                    for i in G.nodes:
                        pos[i] = Wa[:, i].numpy()
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
                plt.savefig(f"{file_path[:-3]}_{k}.png")
                plt.show()
                plt.clf()
                plt.close()
                gc.collect()

            for i in range(N):
                idx = s[:, 0] == i
                idx = np.argwhere(idx)
                if len(idx) > 0:
                    for q in idx:
                        for w in idx:
                            A_samples[q[0], w[0]] += 1

        G = nx.Graph()
        we = []
        for i in range(0, A_samples.shape[0]):
            for j in range(i + 1, A_samples.shape[1]):
                if A_samples[i, j] > 0:
                    we.append((i, j, A_samples[i, j]))
        G.add_weighted_edges_from(we)
        w = []
        for e in G.edges:
            w.append(A_samples[e[0], e[1]])
        wd = np.array(w)
        widths = MinMaxScaler(feature_range=(0.1, 1)).fit_transform(wd.reshape(-1, 1)).squeeze().tolist()

        node_colors_list = []
        for node in G.nodes:
            node_colors_list.append(cmap[y[node]])

        pos = nx.drawing.layout.spring_layout(G, seed=42)
        plt.figure(figsize=[5, 4])
        c = '#00838F'
        nx.draw_networkx_nodes(G, pos=pos, node_size=10, node_color=node_colors_list)
        nx.draw_networkx_edges(G, pos=pos, width=widths, edge_color=c)
        plt.title(title)
        plt.savefig(f"{file_path[:-4]}_samples.png")
        plt.show()
        plt.clf()
        plt.close()
        gc.collect()

        self.A_samples_ = A_samples


def _squared_dist(A, B):
    row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
    row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

    row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
    row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

    return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B
