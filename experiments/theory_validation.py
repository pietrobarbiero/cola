import gc
import sys
import os

from sklearn import clone, datasets
from sklearn.cluster import k_means, KMeans
from sklearn.datasets import make_circles, make_moons, make_blobs, make_classification, load_digits
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging
import tensorflow as tf

from tensorflow.keras import Input
from tqdm import tqdm

from cola import BaseModel, DualModel, quantization
from cola._utils import score, compute_graph, scatterplot, scatterplot_dynamic, dynamic_decay


def _squared_dist(A, B):
    row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
    row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

    row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
    row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

    return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B


def main():

    results_dir = "./paper0"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    logging.basicConfig(filename=os.path.join(results_dir, 'results.log'),
                        level=logging.INFO)

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

    theta = np.radians(np.linspace(30, 360 * 4, n_samples))
    r = theta ** 2
    x_2 = r * np.cos(theta)
    y_2 = r * np.sin(theta)
    X_spiral = np.vstack([x_2, y_2]).T
    y_spiral = np.zeros(len(X_spiral))
    # plt.figure()
    # plt.scatter(x_2, y_2)
    # plt.show()

    Xl, yl = make_classification(n_samples=n_samples, n_features=2, class_sep=25,
                                 n_informative=2, n_redundant=0, hypercube=False,
                                 n_classes=4, n_clusters_per_class=1, random_state=42)
    Xh, yh = make_classification(n_samples=n_samples, n_features=3000, class_sep=8,
                                 n_informative=10, n_redundant=2990, hypercube=False,
                                 n_classes=4, n_clusters_per_class=1, random_state=42)

    x_digits, y_digits = load_digits(return_X_y=True)

    datasets = {
        # "digits": (x_digits, y_digits),
        # "Spiral": (X_spiral, y_spiral),
        # "Circles": noisy_circles,
        "Moons": noisy_moons,
        # "Blobs (low)": (Xl, yl),
        # "Blobs (high)": (Xh, yh),
        # "gabri": (X_gabri, y_gabri),
        # "Ellipsoids": aniso,
        # "Blobs": blobs,
        # "varied": varied,
    }

    bar_position = 0
    progress_bar = tqdm(datasets.items(), position=bar_position)
    for dataset, data in progress_bar:
        progress_bar.set_description("Analysis of dataset: %s" % dataset)
        X, y = data
        X = StandardScaler().fit_transform(X)

        # u, s, vh = np.linalg.svd(X)
        # print(f'dataset: {dataset} | max s: {np.max(s)} - min s: {np.min(s)}')
        # continue

        ns, nf = X.shape
        k = 40
        epochs = 800
        lr_dual = 0.000008
        lr_base = 0.008
        lmb_dual = 0.01
        lmb_standard = 0.01
        repetitions = 1

        kmeans_losses = []
        mlp_losses = []
        mlp_loss_Q = []
        mlp_loss_E = []
        mlp_time = []
        mlp_nodes = []
        trans_losses = []
        trans_loss_Q = []
        trans_loss_E = []
        trans_time = []
        trans_nodes = []
        steps = []
        for i in range(repetitions):
            inputs = Input(shape=(nf,), name='input')
            vanilla = BaseModel(n_features=nf, k_prototypes=k, deep=False, inputs=inputs, outputs=inputs)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_base)
            vanilla.compile(optimizer=optimizer)
            # model.layers[1].summary()
            vanilla.fit(X, y, epochs=epochs, verbose=False)
            prototypes = vanilla.base_model.weights[-1].numpy()
            # plt.figure(figsize=[4, 3])
            # dynamic_decay(X, model.prototypes_[:200], dim=0, valid=True, scale='log')
            # plt.savefig(f'{dataset}_decayX_vanilla.pdf')
            # plt.savefig(f'{dataset}_decayX_vanilla.png')
            # plt.show()
            # plt.figure(figsize=[4, 3])
            # dynamic_decay(X, vanilla.prototypes_[:40], valid=True, scale='log')
            # plt.savefig(f'{dataset}_decayY_vanilla.pdf')
            # plt.savefig(f'{dataset}_decayY_vanilla.png')
            # plt.show()
            # plt.figure()
            # scatterplot_dynamic(X, model.prototypes_, y, valid=True)
            # plt.savefig(f'{dataset}_dynamic_vanilla.pdf')
            # plt.savefig(f'{dataset}_dynamic_vanilla.png')
            # plt.show()

            # Dual
            print("Dual Model")
            inputs = Input(shape=(nf,), name='input')
            model = DualModel(n_samples=ns, k_prototypes=k, deep=False, inputs=inputs, outputs=inputs)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_dual)
            model.compile(optimizer=optimizer)
            model.fit(X, y, epochs=epochs, verbose=False)
            x_pred = model.predict(X)
            prototypes = model.dual_model.predict(x_pred.T)
            # plt.figure(figsize=[4, 3])
            # dynamic_decay(X, model.prototypes_[:200], dim=0, valid=True, scale='log')
            # plt.savefig(f'{dataset}_decayX_dual.pdf')
            # plt.savefig(f'{dataset}_decayX_dual.png')
            # plt.show()
            plt.figure(figsize=[4, 3])
            dynamic_decay(X, vanilla.prototypes_, is_dual=False, valid=True, scale='log', c='b')
            dynamic_decay(X, model.prototypes_, valid=True, scale='log', c='r')
            plt.savefig(f'{dataset}_decayY_dual.pdf')
            plt.savefig(f'{dataset}_decayY_dual.png')
            plt.show()
            # plt.figure()
            # scatterplot_dynamic(X, model.prototypes_, y, valid=True)
            # plt.savefig(f'{dataset}_dynamic_dual.pdf')
            # plt.savefig(f'{dataset}_dynamic_dual.png')
            # plt.show()


if __name__ == "__main__":
    sys.exit(main())
