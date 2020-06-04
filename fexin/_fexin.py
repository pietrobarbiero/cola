import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import networkx as nx
from tqdm import tqdm


class Fexin():
    def __init__(self, kmodel, E=None, optimizer=None, verbose=True):
        self.kmodel = kmodel
        self.E = E
        self.optimizer = optimizer
        self.verbose = verbose

    def fit(self, X, N=None, num_epochs=200):
        self.A_ = tf.convert_to_tensor(X.T, np.float32)
        if N is None:
            N = 3

        if self.E is None:
            self.E_ = np.zeros((N, N))
        else:
            self.E_ = self.E

        if self.optimizer is None:
            self.optimizer_ = tf.keras.optimizers.Adam(learning_rate=0.01)
        else:
            self.optimizer_ = self.optimizer

        # pbar = tqdm(range(num_epochs+1), disable=self.verbose)
        for epoch in range(num_epochs+1):
            loss_value, grads, Ei = _grad(self.kmodel, self.A_, self.E_)
            self.optimizer_.apply_gradients(zip(grads, self.kmodel.trainable_variables))
            # if self.verbose:
            #     pbar.set_description(f"Epoch: {epoch} - Loss: {loss_value:.2f}")
        self.Ei_ = Ei
        self.loss_value_ = loss_value
        return self

    def predict(self, X):
        return self.kmodel(X).numpy().T

    def plot(self, X, title, file_path):
        Ga = nx.from_numpy_matrix(self.Ei_)
        Wa = self.predict(self.A_)

        pos = {}
        for i in range(Wa.shape[0]):
            pos[i] = Wa[i]

        if X.shape[1] > 2:
            Xp = TSNE().fit_transform(X)
        else:
            Xp = X

        plt.figure(figsize=[5, 4])
        sns.scatterplot(Xp[:, 0], Xp[:, 1])
        nx.draw_networkx(Ga, pos=pos, node_size=600, alpha=0.8, node_color='red')
        plt.title(title)
        plt.savefig(file_path)
        plt.show()


def _squared_dist(A, B):
    row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
    row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

    row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
    row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

    return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B


def _loss(X, output, Et):
    A = tf.convert_to_tensor(tf.transpose(X), np.float32)
    D = _squared_dist(A, tf.transpose(output))
    d = tf.math.reduce_min(D, axis=1)

    s = tf.argsort(D.numpy(), axis=1)[:, :2].numpy()
    W = tf.sort(D, axis=1)[:, :2].numpy()

    for i in range(s.shape[0]):
        Et[s[i, 0], s[i, 1]] += 1
        Et[s[i, 1], s[i, 0]] += 1

    E = tf.convert_to_tensor(Et, np.float32)

    return tf.norm(d) + tf.norm(E), Et


def _grad(model, inputs, Et):
    with tf.GradientTape() as tape:
        loss_value, Et = _loss(inputs, model(inputs), Et)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), Et
