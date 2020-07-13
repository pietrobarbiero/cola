import sys
import os

from keras.datasets.mnist import load_data
from sklearn import clone, datasets
from sklearn.datasets import make_circles, make_moons, make_blobs, load_digits
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from deeptl import DeepTopologicalClustering


def main():
    results_dir = "results"
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
        "digits": (x_digits, y_digits),
        "mnist": (x_test, y_test),
        "gabri": (X_gabri, y_gabri),
        "noisy_circles": noisy_circles,
        "noisy_moons": noisy_moons,
        "blobs": blobs,
        "aniso": aniso,
        "varied": varied,
    }
    # for dataset, data in datasets.items():
    #     d = pd.DataFrame(data)
    #     d.to_csv(f"{dataset}.csv")
    # return

    bar_position = 0
    progress_bar = tqdm(datasets.items(), position=bar_position)
    for dataset, data in progress_bar:
        progress_bar.set_description("Analysis of dataset: %s" % dataset)
        X, y = data

        if dataset == 'mnist':
            X2 = X / 255
            X3 = np.expand_dims(X2, axis=3)
            X4 = np.append(X3, X3, axis=3)
            X4 = np.append(X4, X3, axis=3)
            XZ = np.zeros((X4.shape[0], 32, 32, 3))
            XZ[:, 2:30, 2:30, :] = X4
            IMG_SHAPE = (32, 32, 3)
            base_model = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE,
                                                           include_top=False,
                                                           weights='imagenet')
            preds = base_model.predict(XZ)
            preds = np.reshape(preds, (preds.shape[0], preds.shape[-1]))
            X = preds

        X = StandardScaler().fit_transform(X)

        N = 500
        model = DeepTopologicalClustering(verbose=False, N=N, num_epochs=400, lr=0.0008)
        model.fit(X)
        accuracy = model.score(y)

        print(f'Accuracy: {accuracy:.4f}')

        title = f'Accuracy: {accuracy:.4f}'
        model.plot_confusion_matrix(y, title, os.path.join(results_dir, f"{dataset}_confmat.pdf"))

        # model.compute_sample_graph()
        # model.compute_graph()
        # model.plot_adjacency_matrix()
        # model.plot_graph(y, os.path.join(results_dir, f"{dataset}.pdf"))
        # model.plot_sample_graph(y, os.path.join(results_dir, f"{dataset}_samples.pdf"))
        # model.plot_graph(y, os.path.join(results_dir, f"{dataset}.png"))
        # model.plot_sample_graph(y, os.path.join(results_dir, f"{dataset}_samples.png"))
        # pd.DataFrame(model.adjacency_matrix_).to_csv(os.path.join(results_dir, f"{dataset}.csv"))
        # pd.DataFrame(model.centroids_).to_csv(os.path.join(results_dir, f"{dataset}_centroids.csv"))
        # pd.DataFrame(model.adjacency_samples_).to_csv(os.path.join(results_dir, f"{dataset}_samples.csv"))


if __name__ == "__main__":
    sys.exit(main())
