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

from deeptl import DeepTopologicalNonstationaryClustering, DeepCompetitiveLayerNonStationary


def main():

    # results_dir = "./nonstationary"
    results_dir = "./prova"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    logging.basicConfig(filename=os.path.join(results_dir, 'results.log'),
                        level=logging.INFO)

    n_samples = 200
    x_square = np.random.uniform(0, 1, (n_samples, 2))
    x_square_1 = x_square.copy()
    x_square_1[:, 1] = x_square[:, 1] + 2
    x_square_2 = x_square.copy()
    x_square_2[:, 0] = x_square[:, 0] + 2
    x_square_3 = x_square.copy()
    x_square_3[:, 0] = x_square[:, 0] + 2
    x_square_3[:, 1] = x_square[:, 1] + 2
    x_square = np.concatenate([x_square, x_square_1, x_square_2, x_square_3])
    y_square = np.zeros((x_square.shape[0],), dtype='int')
    L = len(x_square_1)
    y_square[L:2*L] = 1
    y_square[2*L:3*L] = 2
    y_square[3*L:4*L] = 3

    # plt.figure()
    # plt.scatter(x_square[:, 0], x_square[:, 1], c=y_square)
    # plt.show()
    # return

    datasets = {
        "square": (x_square, y_square),
    }

    bar_position = 0
    progress_bar = tqdm(datasets.items(), position=bar_position)
    for dataset, data in progress_bar:
        progress_bar.set_description("Analysis of dataset: %s" % dataset)
        X, y = data
        X = StandardScaler().fit_transform(X)

        N = 30
        num_epochs = 400
        lr_dual = 0.0008
        lr_standard = 0.008
        lmb_dual = 0.01
        lmb_standard = 0.01
        repetitions = 10

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
            model_mlp = DeepCompetitiveLayerNonStationary(verbose=False, lmb=lmb_standard,
                                             N=N, num_epochs=num_epochs, lr=lr_standard)
            start_time = time.time()
            model_mlp.fit(X, y)
            mlp_time.append(time.time() - start_time)
            model_mlp.compute_graph()
            model_mlp.plot_graph(X, y, os.path.join(results_dir, f"{dataset}_{i}_standard.pdf"))
            model_mlp.plot_graph(X, y, os.path.join(results_dir, f"{dataset}_standard.png"))
            mlp_losses.extend(model_mlp.loss_vals)
            mlp_loss_Q.extend(model_mlp.loss_Q_)
            mlp_loss_E.extend(model_mlp.loss_E_)
            mlp_nodes.extend(model_mlp.node_list_)

            model_trans = DeepTopologicalNonstationaryClustering(verbose=False, lmb=lmb_dual,
                                                                 N=N, num_epochs=num_epochs, lr=lr_dual)
            start_time = time.time()
            model_trans.fit(X, y)
            trans_time.append(time.time() - start_time)
            model_trans.compute_graph()
            model_trans.plot_graph(X, y, os.path.join(results_dir, f"{dataset}_{i}_dual.pdf"))
            model_trans.plot_graph(X, y, os.path.join(results_dir, f"{dataset}_dual.png"))
            trans_losses.extend(model_trans.loss_vals)
            trans_loss_Q.extend(model_trans.loss_Q_)
            trans_loss_E.extend(model_trans.loss_E_)
            trans_nodes.extend(model_trans.node_list_)

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
        })

        sns.set_style('whitegrid')
        plt.figure(figsize=[4, 3])
        sns.lineplot('epoch', 'standard', data=losses_Q, label='standard', ci=99)
        sns.lineplot('epoch', 'dual', data=losses_Q, label='dual', ci=99)
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


        # nodes = pd.DataFrame({
        #     'epoch': steps,
        #     'standard': mlp_nodes,
        #     'dual': trans_nodes,
        # })
        #
        # sns.set_style('whitegrid')
        # plt.figure(figsize=[4, 3])
        # sns.lineplot('epoch', 'standard', data=nodes, label='standard', ci=99)
        # sns.lineplot('epoch', 'dual', data=nodes, label='dual', ci=99)
        # # plt.yscale('log')
        # plt.ylabel('number of prototypes')
        # plt.title(f'{dataset}')
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
        # plt.tight_layout()
        # plt.savefig(os.path.join(results_dir, f'{dataset}_nodes.png'))
        # plt.savefig(os.path.join(results_dir, f'{dataset}_nodes.pdf'))
        # plt.show()


        logging.info(f'Analyzing dataset: {dataset}')
        logging.info(f'Standard competitive layer elapsed time: {np.mean(mlp_time):.2f} +- {np.std(mlp_time):.2f}')
        logging.info(f'Dual layer elapsed time: {np.mean(trans_time):.2f} +- {np.std(trans_time):.2f}')


if __name__ == "__main__":
    sys.exit(main())
