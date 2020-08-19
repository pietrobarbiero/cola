import sys
import os

from keras.datasets.mnist import load_data
from sklearn import clone, datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_circles, make_moons, make_blobs, load_digits
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.utils.vis_utils import plot_model
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model

from cole import DualModel, quantization, plot_confusion_matrix, \
    scatterplot, compute_graph, BaseModel


def main():
    results_dir = "cole"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    n_samples = 30
    noisy_circles = make_circles(n_samples=n_samples, factor=.5,
                                 noise=.05)
    noisy_moons = make_moons(n_samples=n_samples, noise=.05)
    blobs = make_blobs(n_samples=n_samples, centers=3, random_state=8)
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

    theta = np.radians(np.linspace(30, 360 * 4, n_samples))
    r = theta ** 2
    x_2 = r * np.cos(theta)
    y_2 = r * np.sin(theta)
    X_spiral = np.vstack([x_2, y_2]).T
    y_spiral = np.zeros(len(X_spiral))
    # plt.figure()
    # plt.scatter(x_2, y_2)
    # plt.show()

    (x_train, y_train), (x_test, y_test) = load_data()
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[1]))

    x_digits, y_digits = load_digits(return_X_y=True)

    datasets = {
        # "Spiral": (X_spiral, y_spiral),
        # "digits": (x_digits, y_digits),
        # "mnist": (x_test[:5000], y_test[:5000]),
        # "gabri": (X_gabri, y_gabri),
        # "noisy_circles": noisy_circles,
        # "noisy_moons": noisy_moons,
        "blobs": blobs,
        # "aniso": aniso,
        # "varied": varied,
    }

    bar_position = 0
    progress_bar = tqdm(datasets.items(), position=bar_position)
    for dataset, data in progress_bar:
        progress_bar.set_description("Analysis of dataset: %s" % dataset)
        X, y = data

        # if dataset == 'mnist':
        #     mnist_file = os.path.join(results_dir, 'mnist.csv')
        #     if not os.path.isfile(mnist_file):
        #         X2 = X / 255
        #         X3 = np.expand_dims(X2, axis=3)
        #         X4 = np.append(X3, X3, axis=3)
        #         X4 = np.append(X4, X3, axis=3)
        #         XZ = np.zeros((X4.shape[0], 32, 32, 3))
        #         XZ[:, 2:30, 2:30, :] = X4
        #         IMG_SHAPE = (32, 32, 3)
        #         pretrained_model = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE,
        #                                                      include_top=False,
        #                                                      weights='imagenet')
        #         # Unfreeze the base model
        #         pretrained_model.trainable = True
        #         inputs = Input(shape=IMG_SHAPE)
        #         x = pretrained_model(inputs, training=True)
        #         x = GlobalAveragePooling2D()(x)
        #         outputs = Dense(len(set(y)))(x)
        #         pre_model = Model(inputs, outputs)
        #         pre_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Very low learning rate
        #                       loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        #                       metrics=[tf.keras.metrics.CategoricalAccuracy()])
        #
        #         # Train end-to-end. Be careful to stop before you overfit!
        #         pre_model.fit(XZ, to_categorical(y), epochs=70, batch_size=200)
        #
        #         preds = pretrained_model.predict(XZ)
        #         X = np.reshape(preds, (preds.shape[0], preds.shape[-1]))
        #         Xpd = pd.DataFrame(X)
        #         Xpd.to_csv(mnist_file)
        #
        #     else:
        #         X = pd.read_csv(mnist_file, index_col=0).values
        #
        #     # # Freeze base model
        #     # pretrained_model.trainable = False
        #     # # Create new model on top.
        #     # inputs = Input(shape=IMG_SHAPE)
        #     # x = pretrained_model(inputs, training=False)
        #     # x = GlobalAveragePooling2D()(x)
        #     # outputs = Flatten()(x)
        #     # X = XZ

        X = StandardScaler().fit_transform(X)

        n = X.shape[0]
        d = X.shape[1]
        latent_dim = 2
        k = 3
        lr = 0.008
        epochs = 300
        lbd = 0.01

        # inputs = Input(shape=(d,), name='input')
        # outputs = inputs
        # model = BaseModel(n_features=d, k_prototypes=k, inputs=inputs, outputs=outputs, deep=False)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=0.008)
        # model.compile(optimizer=optimizer)
        # model.summary()
        # model.fit(X, y, epochs=epochs, verbose=True)
        # x_pred = model.predict(X)
        # prototypes = model.base_model.weights[-1].numpy()
        # # G = compute_graph(x_pred, prototypes)
        # # plt.figure()
        # # plot_confusion_matrix(x_pred, prototypes, y)
        # # plt.show()
        # plt.figure()
        # scatterplot(x_pred, prototypes, y, valid=False)
        # plt.savefig(os.path.join(results_dir, dataset + ".png"))
        # plt.show()

        inputs = Input(shape=(d,), name='input')
        outputs = inputs
        # x = Dense(512)(inputs)
        # outputs = Dense(256)(x)
        # x = Dense(128)(x)
        # outputs = Dense(64)(x)
        model = DualModel(n_samples=n, k_prototypes=k, inputs=inputs, outputs=outputs, deep=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer)
        model.summary()
        model.fit(X, y, epochs=epochs)
        x_pred = model.predict(X)
        prototypes = model.dual_model.predict(x_pred.T)
        G = compute_graph(x_pred, prototypes)
        # plt.figure()
        # plot_confusion_matrix(x_pred, prototypes, y)
        # plt.show()
        plt.figure()
        scatterplot(x_pred, prototypes, y, valid=True)
        plt.show()

        # k1 = len(G.nodes)
        # k_means = KMeans(n_clusters=k1)
        # k_means.fit(x_pred)
        # # plt.figure()
        # # plot_confusion_matrix(x_pred, k_means.cluster_centers_.T, y)
        # # plt.show()
        # plt.figure()
        # scatterplot(x_pred, k_means.cluster_centers_.T, y, links=False)
        # plt.show()


if __name__ == "__main__":
    sys.exit(main())
