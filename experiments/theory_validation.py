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

from deeptl import DeepCompetitiveLayer, DeepTopologicalClustering


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
        "Spiral": (X_spiral, y_spiral),
        "Circles": noisy_circles,
        "Moons": noisy_moons,
        "Blobs (low)": (Xl, yl),
        "Blobs (high)": (Xh, yh),
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

        N = 40
        num_epochs = 400
        lr_dual = 0.0008
        lr_standard = 0.008
        lmb_dual = 0.01
        lmb_standard = 0.01
        repetitions = 5

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
            model_mlp = DeepCompetitiveLayer(verbose=False, lmb=lmb_standard,
                                             N=N, num_epochs=num_epochs, lr=lr_standard)
            start_time = time.time()
            model_mlp.fit(X)
            mlp_time.append(time.time() - start_time)
            # model_mlp.compute_sample_graph()
            model_mlp.compute_graph()
            # model_mlp.plot_adjacency_matrix()
            model_mlp.plot_graph(y, os.path.join(results_dir, f"{dataset}_{i}_standard.pdf"))
            model_mlp.plot_graph(y, os.path.join(results_dir, f"{dataset}_standard.png"))
            # model_mlp.plot_sample_graph(y, os.path.join(results_dir, f"{dataset}_samples_standard.pdf"))
            mlp_losses.extend(model_mlp.loss_vals)
            mlp_loss_Q.extend(model_mlp.loss_Q_)
            mlp_loss_E.extend(model_mlp.loss_E_)
            mlp_nodes.extend(model_mlp.node_list_)

            # for iter in range(num_epochs+1):
            #     # model_km = KMeans(n_clusters=model_trans.node_list_[-1], max_iter=num_epochs).fit(X)
            #     model_km = KMeans(n_clusters=N, n_init=5, max_iter=num_epochs, random_state=42).fit(X)
            #     D = _squared_dist(tf.Variable(X), tf.Variable(model_km.cluster_centers_))
            #     d_min = tf.math.reduce_min(D, axis=1)
            #     loss = tf.norm(d_min).numpy()
            #     kmeans_losses.append(loss)
            #     print(f'K-means: iteration {iter}/{num_epochs} | loss: {loss:.4f}')
            # # plt.figure(figsize=[5, 4])
            # # fig, ax = plt.subplots()
            # # cmap = sns.color_palette(sns.color_palette("hls", len(set(y))))
            # # sns.scatterplot(X[:, 0], X[:, 1], hue=y, palette=cmap, hue_order=set(y), alpha=0.3, legend=False)
            # # c = '#00838F'
            # # plt.scatter(model_km.cluster_centers_[:, 0], model_km.cluster_centers_[:, 1], c=c, s=200)
            # # ax.axis('off')
            # # plt.tight_layout()
            # # plt.savefig(os.path.join(results_dir, f"{dataset}_{i}_kmeans.pdf"))
            # # plt.savefig(os.path.join(results_dir, f"{dataset}_{i}_kmeans.png"))
            # # plt.show()
            # # plt.clf()
            # # plt.close()
            # # gc.collect()

            model_trans = DeepTopologicalClustering(verbose=False, lmb=lmb_dual,
                                                    N=N, num_epochs=num_epochs, lr=lr_dual)
            start_time = time.time()
            model_trans.fit(X)
            trans_time.append(time.time() - start_time)
            # model_trans.compute_sample_graph()
            model_trans.compute_graph()
            # model_trans.plot_adjacency_matrix()
            model_trans.plot_graph(y, os.path.join(results_dir, f"{dataset}_{i}_dual.pdf"))
            model_trans.plot_graph(y, os.path.join(results_dir, f"{dataset}_dual.png"))
            # model_trans.plot_sample_graph(y, os.path.join(results_dir, f"{dataset}_samples_dual.pdf"))
            trans_losses.extend(model_trans.loss_vals)
            trans_loss_Q.extend(model_trans.loss_Q_)
            trans_loss_E.extend(model_trans.loss_E_)
            trans_nodes.extend(model_trans.node_list_)

            # return

            steps.extend(np.arange(0, len(model_trans.loss_vals)))

        losses = pd.DataFrame({
            'epoch': steps,
            'standard': mlp_losses,
            'dual': trans_losses,
        })

        sns.set_style('whitegrid')
        plt.figure(figsize=[4, 3])
        sns.lineplot('epoch', 'standard', data=losses, label='standard', ci=99)
        sns.lineplot('epoch', 'dual', data=losses, label='dual', ci=99)
        # plt.yscale('log')
        plt.ylabel('loss')
        plt.title(f'{dataset}')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{dataset}_loss.png'))
        plt.savefig(os.path.join(results_dir, f'{dataset}_loss.pdf'))
        plt.show()


        losses_Q = pd.DataFrame({
            'epoch': steps,
            'standard': mlp_loss_Q,
            'dual': trans_loss_Q,
            # 'kmeans': kmeans_losses,
        })

        sns.set_style('whitegrid')
        plt.figure(figsize=[4, 3])
        sns.lineplot('epoch', 'standard', data=losses_Q, label='standard', ci=99)
        sns.lineplot('epoch', 'dual', data=losses_Q, label='dual', ci=99)
        # sns.lineplot('epoch', 'kmeans', data=losses_Q, label='kmeans', ci=99)
        # plt.yscale('log')
        plt.ylabel('quantization error')
        plt.title(f'{dataset}')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{dataset}_loss_q.png'))
        plt.savefig(os.path.join(results_dir, f'{dataset}_loss_q.pdf'))
        plt.show()


        losses_E = pd.DataFrame({
            'epoch': steps,
            'standard': mlp_loss_E,
            'dual': trans_loss_E,
        })

        sns.set_style('whitegrid')
        plt.figure(figsize=[4, 3])
        sns.lineplot('epoch', 'standard', data=losses_E, label='standard', ci=99)
        sns.lineplot('epoch', 'dual', data=losses_E, label='dual', ci=99)
        # plt.yscale('log')
        plt.ylabel('topological complexity')
        plt.title(f'{dataset}')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{dataset}_loss_e.png'))
        plt.savefig(os.path.join(results_dir, f'{dataset}_loss_e.pdf'))
        plt.show()


        nodes = pd.DataFrame({
            'epoch': steps,
            'standard': mlp_nodes,
            'dual': trans_nodes,
        })

        sns.set_style('whitegrid')
        plt.figure(figsize=[4, 3])
        sns.lineplot('epoch', 'standard', data=nodes, label='standard', ci=99)
        sns.lineplot('epoch', 'dual', data=nodes, label='dual', ci=99)
        # plt.yscale('log')
        plt.ylabel('number of prototypes')
        plt.title(f'{dataset}')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{dataset}_nodes.png'))
        plt.savefig(os.path.join(results_dir, f'{dataset}_nodes.pdf'))
        plt.show()


        logging.info(f'Analyzing dataset: {dataset}')
        logging.info(f'Standard competitive layer elapsed time: {np.mean(mlp_time):.2f} +- {np.std(mlp_time):.2f}')
        logging.info(f'Dual layer elapsed time: {np.mean(trans_time):.2f} +- {np.std(trans_time):.2f}')


if __name__ == "__main__":
    sys.exit(main())
