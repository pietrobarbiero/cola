import sys
import os

from keras.datasets.mnist import load_data
from sklearn import clone, datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_circles, make_moons, make_blobs, load_digits
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.utils.vis_utils import plot_model
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model

from cole import DualModel, qe_loss, plot_confusion_matrix


def main():
    results_dir = "cole"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    n_samples = 500
    noisy_circles = make_circles(n_samples=n_samples, factor=.5,
                                 noise=.05)
    noisy_moons = make_moons(n_samples=n_samples, noise=.05)
    blobs = make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)
    varied = make_blobs(n_samples=n_samples,
                        cluster_std=[1.0, 2.5, 0.5],
                        random_state=random_state)

    y = np.asarray([i for i in range(1000)])
    x = np.asarray([i % 4 for i in range(1000)])
    X_gabri = np.vstack([x, y]).T
    y_gabri = x
    # plt.figure()
    # sns.scatterplot(X_gabri[:, 0], X_gabri[:, 1], hue=y)
    # plt.show()

    (x_train, y_train), (x_test, y_test) = load_data()
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[1]))

    x_digits, y_digits = load_digits(return_X_y=True)

    datasets = {
        # "digits": (x_digits, y_digits),
        # "mnist": (x_test, y_test),
        # "gabri": (X_gabri, y_gabri),
        # "noisy_circles": noisy_circles,
        "noisy_moons": noisy_moons,
        # "blobs": blobs,
        # "aniso": aniso,
        # "varied": varied,
    }

    bar_position = 0
    progress_bar = tqdm(datasets.items(), position=bar_position)
    for dataset, data in progress_bar:
        progress_bar.set_description("Analysis of dataset: %s" % dataset)
        X, y = data
        X = StandardScaler().fit_transform(X)

        n = X.shape[0]
        d = X.shape[1]
        latent_dim = 2
        k = 30
        lr = 0.0008
        lbd = 0.01

        inputs = Input(shape=(d,), name='input')
        outputs = inputs
        x = Dense(32)(inputs)
        # x = Dense(128)(x)
        # x = Dense(256)(x)
        # outputs = Dense(512)(x)
        # model = DualModel(n_samples=n, k_prototypes=k, inputs=inputs, outputs=outputs)
        model = DualModel(n_samples=n, k_prototypes=k, inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer)
        model.summary()
        model.fit(X, X, epochs=100)
        x_pred = model.predict(X)
        prototypes = model.dual_model.predict(x_pred.T)

        plot_confusion_matrix(x_pred, prototypes, y,
                              file_name=os.path.join(results_dir, f"{dataset}_confmat_dual.pdf"))

        tsne = TSNE(n_components=2, random_state=42)
        M_list = [x_pred]
        for i, node in enumerate(prototypes.T):
            M_list.append(node.reshape(1, -1))
        M = np.concatenate(M_list)
        Mp = tsne.fit_transform(M)
        Xp = Mp[:X.shape[0]]
        Wp = Mp[X.shape[0]:]
        plt.figure()
        # plt.scatter(X[:, 0], X[:, 1])
        plt.scatter(Xp[:, 0], Xp[:, 1], c=y)
        plt.scatter(Wp[:, 0], Wp[:, 1], s=100, c='orange')
        plt.show()

        k_means = KMeans(n_clusters=k)
        k_means.fit(x_pred)
        plot_confusion_matrix(x_pred, k_means.cluster_centers_.T, y,
                              file_name=os.path.join(results_dir, f"{dataset}_confmat_kmeans.pdf"))


if __name__ == "__main__":
    sys.exit(main())
