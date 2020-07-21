import collections
import gc
import sys
import os

from sklearn import clone, datasets
from sklearn.cluster import k_means, KMeans
from sklearn.datasets import make_circles, make_moons, make_blobs, make_classification, load_digits
from sklearn.decomposition import PCA
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


def compute_confusion_matrix(model, X, y, N):
    D = _squared_dist(tf.Variable(X), tf.Variable(model.cluster_centers_))
    s = tf.argmin(D.numpy(), axis=1).numpy()
    sz = len(set(y))
    confusion_matrix = np.zeros((sz, sz))
    for i in range(N):
        idx = s == i
        if sum(idx) > 0:
            counts = collections.Counter(y[idx])
            km, vm = counts.most_common(1)[0]
            for k, v in counts.items():
                confusion_matrix[km, k] += v
    return confusion_matrix


def plot_confusion_matrix(confmat, title='', file_name=None, figsize=[5, 5], show=True):
    score = sum(np.diag(confmat)) / sum(sum(confmat))
    title = f'Accuracy: {score:.4f}'
    plt.figure(figsize=figsize)
    sns.heatmap(confmat.astype('int'), annot=True, fmt='d',
                cbar=False, square=True, cmap='Greens')
    plt.title(title)
    plt.ylabel('true')
    plt.xlabel('predicted')
    plt.tight_layout()
    plt.savefig(file_name)
    if show:
        plt.show()
    else:
        plt.close()
    return


def main():
    results_dir = "./dimensionality5"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    logging.basicConfig(filename=os.path.join(results_dir, 'results.log'),
                        level=logging.INFO)

    experiments = {}
    n_samples = [100, 1000, 10000]
    n_informative = [1.0, 0.5]
    n_features = [100, 300, 1000, 3000, 10000]  # 100000
    for ns in n_samples:
        # for nf in n_features:
            for ni in n_informative:
                # if ni <= nf:
                #     print(f'{nf} {ni}')
                    experiments[f's_{ns}_i_{ni}'] = [ns, ni]

    list_acc_base = []
    list_acc_dual = []
    list_acc_kmeans = []
    ns_count = []
    nf_count = []
    ni_count = []
    ds_name = []
    bar_position = 0
    progress_bar = tqdm(experiments.items(), position=bar_position)
    for j, (dataset, data) in enumerate(progress_bar):
        progress_bar.set_description("Analysis of dataset: %s" % dataset)
        ns, ni = data
        progress_bar_2 = tqdm(n_features, position=1)
        for nf in progress_bar_2:

            # Xc = np.matmul(X, X.T)
            # V = np.linalg.eig(Xc)[1]
            # np.matmul(V.T, X).shape

            # plt.figure()
            # plt.plot(np.arange(len(E[0])), E[0])
            # plt.show()
            # return

            ni2 = int(nf * ni)
            N = int(ns/20)
            # num_epochs = 1
            num_epochs = 300
            lr_dual = 0.008
            lr_base = 0.008
            lmb_dual = 0  # 0.01
            lmb_base = 0  # 0.01
            repetitions = 10
            # repetitions = 1

            acc_base = []
            acc_dual = []
            acc_kmeans = []
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
            acc_dbscan = []
            acc_rf = []
            steps = []
            progress_bar_3 = tqdm(range(repetitions), position=1)
            for i in progress_bar_3:
                X, y = make_classification(n_samples=ns, n_features=nf, class_sep=8,
                                           n_informative=ni2, n_redundant=0, hypercube=True,
                                           n_classes=2, n_clusters_per_class=1, random_state=i)
                X = StandardScaler().fit_transform(X)

                model_mlp = DeepCompetitiveLayer(verbose=False, lmb=lmb_base,
                                                 N=N, num_epochs=num_epochs, lr=lr_base)
                start_time = time.time()
                model_mlp.fit(X)
                mlp_time.append(time.time() - start_time)
                accuracy = model_mlp.score(y)
                acc_base.append(accuracy)
                title = f'Accuracy: {accuracy:.4f}'
                # model_mlp.plot_confusion_matrix(y, title, os.path.join(results_dir, f"{dataset}_f_{nf}_{i}_confmat_base.pdf"), show=False)
                # model_mlp.compute_graph()
                # model_mlp.plot_graph(y, os.path.join(results_dir, f"{dataset}_base.png"))
                mlp_losses.extend(model_mlp.loss_vals)
                mlp_loss_Q.extend(model_mlp.loss_Q_)
                mlp_loss_E.extend(model_mlp.loss_E_)
                mlp_nodes.extend(model_mlp.node_list_)

                model_trans = DeepTopologicalClustering(verbose=False, lmb=lmb_dual,
                                                        N=N, num_epochs=num_epochs, lr=lr_dual)
                start_time = time.time()
                model_trans.fit(X)
                trans_time.append(time.time() - start_time)
                accuracy = model_trans.score(y)
                acc_dual.append(accuracy)
                title = f'Accuracy: {accuracy:.4f}'
                # model_trans.plot_confusion_matrix(y, title, os.path.join(results_dir, f"{dataset}_f_{nf}_{i}_confmat_dual.pdf"), show=False)
                # model_trans.compute_graph()
                # model_trans.plot_graph(y, os.path.join(results_dir, f"{dataset}_dual.png"))
                trans_losses.extend(model_trans.loss_vals)
                trans_loss_Q.extend(model_trans.loss_Q_)
                trans_loss_E.extend(model_trans.loss_E_)
                trans_nodes.extend(model_trans.node_list_)

                model_km = KMeans(n_clusters=N, init='random', random_state=i).fit(X)
                D = _squared_dist(tf.Variable(X), tf.Variable(model_km.cluster_centers_))
                d_min = tf.math.reduce_min(D, axis=1)
                loss = tf.norm(d_min).numpy()
                kmeans_losses.extend(len(model_trans.loss_Q_) * [loss])
                confmat = compute_confusion_matrix(model_km, X, y, N)
                score = sum(np.diag(confmat)) / sum(sum(confmat))
                acc_kmeans.append(score)
                # plot_confusion_matrix(confmat, file_name=os.path.join(results_dir, f"{dataset}_f_{nf}_{i}_confmat_kmeans.pdf"), show=False)

                steps.extend(np.arange(0, len(model_trans.loss_vals)))
                ns_count.append(ns)
                nf_count.append(nf)
                ni_count.append(ni)
                ds_name.append(f'S {ns} - I {ni}')

            losses_Q = pd.DataFrame({
                'epoch': steps,
                'base': mlp_loss_Q,
                'dual': trans_loss_Q,
                'kmeans': kmeans_losses,
            })

            sns.set_style('whitegrid')
            plt.figure(figsize=[4, 3])
            sns.lineplot('epoch', 'base', data=losses_Q, label='base', ci=99)
            sns.lineplot('epoch', 'dual', data=losses_Q, label='dual', ci=99)
            sns.lineplot('epoch', 'kmeans', data=losses_Q, label='kmeans', ci=99)
            # plt.yscale('log')
            plt.ylabel('Q')
            plt.title(f'{dataset}_f_{nf}')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'{dataset}_f_{nf}_loss_q.png'))
            plt.savefig(os.path.join(results_dir, f'{dataset}_f_{nf}_loss_q.pdf'))
            plt.show()

            # Add accuracies of iterations for current dataset
            list_acc_base += acc_base
            list_acc_dual += acc_dual
            list_acc_kmeans += acc_kmeans

        accuracies = pd.DataFrame({
            'number_sample': ns_count,
            'number_feature': nf_count,
            'number_info_feature': ni_count,
            'base': list_acc_base,
            'dual': list_acc_dual,
            'kmeans': list_acc_kmeans,
        })

        sns.set_style('whitegrid')
        plt.figure(figsize=[4, 3])
        sns.lineplot('number_feature', 'base', data=accuracies, label='base', ci=99)
        sns.lineplot('number_feature', 'dual', data=accuracies, label='dual', ci=99)
        sns.lineplot('number_feature', 'kmeans', data=accuracies, label='kmeans', ci=99)
        plt.xscale('log', basex=10)
        plt.ylabel('Accuracy')
        plt.title(f'Accuracies on Blobs #s: {ns}, %i: {ni*100}')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'accuracies_Blobs_#s-{ns}_%i-{ni*100}.png'))
        plt.savefig(os.path.join(results_dir, f'accuracies_Blobs_#s-{ns}_%i-{ni*100}.pdf'))
        plt.show()

    # method = []
    # method.extend(['base' for _ in acc_base])
    # method.extend(['dual' for _ in acc_dual])
    # method.extend(['k-Means' for _ in acc_kmeans])
    #
    # accuracies = pd.DataFrame({
    #     'dataset': 3*ds_name,
    #     '# samples': 3*ns_count,
    #     '# features': 3*nf_count,
    #     '# informative': 3*ni_count,
    #     '% informative': np.array(3*ni_count)/np.array(3*nf_count),
    #     'accuracy': [*acc_base, *acc_dual, *acc_kmeans],
    #     'method': method,
    # })
    #
    # sns.set_style('whitegrid')
    # plt.figure(figsize=[5, 10])
    # sns.boxplot(x='accuracy', y='dataset', data=accuracies,
    #             hue='method')
    # plt.ylabel('accuracy')
    # plt.title(f'')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.2, -0.1), ncol=3)
    # plt.tight_layout()
    # plt.savefig(os.path.join(results_dir, f'accuracies.png'))
    # plt.savefig(os.path.join(results_dir, f'accuracies.pdf'))
    # plt.show()

    # accuracies = pd.DataFrame({
    #     '# samples': ns_count,
    #     '# features': nf_count,
    #     '# informative': ni_count,
    #     '% informative': np.array(ni_count)/np.array(nf_count),
    #     'base': acc_base,
    #     'dual': acc_dual,
    #     'k-Means': acc_kmeans,
    # })
    #
    # sns.set_style('whitegrid')
    # plt.figure(figsize=[4, 3])
    # sns.scatterplot('# samples', 'base', data=accuracies, label='base', ci=99)
    # sns.scatterplot('# samples', 'dual', data=accuracies, label='dual', ci=99)
    # sns.scatterplot('# samples', 'k-Means', data=accuracies, label='k-Means', ci=99)
    # # plt.yscale('log')
    # plt.ylabel('accuracy')
    # plt.title(f'')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
    # plt.tight_layout()
    # plt.savefig(os.path.join(results_dir, f'accuracies.png'))
    # plt.savefig(os.path.join(results_dir, f'accuracies.pdf'))
    # plt.show()


        # losses = pd.DataFrame({
        #     'epoch': steps,
        #     'base': mlp_losses,
        #     'dual': trans_losses,
        # })
        #
        # sns.set_style('whitegrid')
        # plt.figure(figsize=[4, 3])
        # sns.lineplot('epoch', 'base', data=losses, label='base', ci=99)
        # sns.lineplot('epoch', 'dual', data=losses, label='dual', ci=99)
        # # plt.yscale('log')
        # plt.ylabel('loss')
        # plt.title(f'{dataset}')
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
        # plt.tight_layout()
        # plt.savefig(os.path.join(results_dir, f'{dataset}_loss.png'))
        # plt.savefig(os.path.join(results_dir, f'{dataset}_loss.pdf'))
        # plt.show()
        #
        # losses_Q = pd.DataFrame({
        #     'epoch': steps,
        #     'base': mlp_loss_Q,
        #     'dual': trans_loss_Q,
        #     # 'kmeans': kmeans_losses,
        # })
        #
        # sns.set_style('whitegrid')
        # plt.figure(figsize=[4, 3])
        # sns.lineplot('epoch', 'base', data=losses_Q, label='base', ci=99)
        # sns.lineplot('epoch', 'dual', data=losses_Q, label='dual', ci=99)
        # # sns.lineplot('epoch', 'kmeans', data=losses_Q, label='kmeans', ci=99)
        # # plt.yscale('log')
        # plt.ylabel('Q')
        # plt.title(f'{dataset}')
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
        # plt.tight_layout()
        # plt.savefig(os.path.join(results_dir, f'{dataset}_loss_q.png'))
        # plt.savefig(os.path.join(results_dir, f'{dataset}_loss_q.pdf'))
        # plt.show()
        #
        # losses_E = pd.DataFrame({
        #     'epoch': steps,
        #     'base': mlp_loss_E,
        #     'dual': trans_loss_E,
        # })
        #
        # sns.set_style('whitegrid')
        # plt.figure(figsize=[4, 3])
        # sns.lineplot('epoch', 'base', data=losses_E, label='base', ci=99)
        # sns.lineplot('epoch', 'dual', data=losses_E, label='dual', ci=99)
        # # plt.yscale('log')
        # plt.ylabel('||E||')
        # plt.title(f'{dataset}')
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
        # plt.tight_layout()
        # plt.savefig(os.path.join(results_dir, f'{dataset}_loss_e.png'))
        # plt.savefig(os.path.join(results_dir, f'{dataset}_loss_e.pdf'))
        # plt.show()
        #
        # nodes = pd.DataFrame({
        #     'epoch': steps,
        #     'base': mlp_nodes,
        #     'dual': trans_nodes,
        # })
        #
        # sns.set_style('whitegrid')
        # plt.figure(figsize=[4, 3])
        # sns.lineplot('epoch', 'base', data=nodes, label='base', ci=99)
        # sns.lineplot('epoch', 'dual', data=nodes, label='dual', ci=99)
        # # plt.yscale('log')
        # plt.ylabel('number of prototypes')
        # plt.title(f'{dataset}')
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
        # plt.tight_layout()
        # plt.savefig(os.path.join(results_dir, f'{dataset}_nodes.png'))
        # plt.savefig(os.path.join(results_dir, f'{dataset}_nodes.pdf'))
        # plt.show()
        #
        # logging.info(f'Analyzing dataset: {dataset}')
        # logging.info(f'base competitive layer elapsed time: {np.mean(mlp_time):.2f} +- {np.std(mlp_time):.2f}')
        # logging.info(f'Dual layer elapsed time: {np.mean(trans_time):.2f} +- {np.std(trans_time):.2f}')


if __name__ == "__main__":
    sys.exit(main())
