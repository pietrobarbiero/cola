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
from tensorflow.keras import Input

from cole import BaseModel, DualModel, quantization
from cole._utils import score, compute_graph


def main():
    results_dir = "./dimensionality6"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    logging.basicConfig(filename=os.path.join(results_dir, 'results.log'),
                        level=logging.INFO)

    experiments = {}
    # n_samples = [100, 1000]
    n_samples = [1000]
    # n_informative = [1.0, 0.5]
    n_informative = [1.0]
    # n_features = [100, 300, 1000, 3000, 10000]  # 100000
    n_features = [100, 300, 800]  # 100000
    for ns in n_samples:
        # for nf in n_features:
            for ni in n_informative:
                # if ni <= nf:
                #     print(f'{nf} {ni}')
                    experiments[f's_{ns}_i_{ni}'] = [ns, ni]

    list_acc_base = []
    list_acc_dual = []
    list_acc_deep_base = []
    list_acc_deep_dual = []
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
            # k = int(ns/10)
            k = int(ns/10)
            # epochs = 300
            epochs = 1000
            lr_dual = 0.008
            lr_base = 0.008
            lmb_dual = 0  # 0.01
            lmb_base = 0  # 0.01
            # repetitions = 10
            repetitions = 10

            acc_base = []
            acc_dual = []
            acc_kmeans = []
            kmeans_losses = []
            base_loss_Q = []
            deep_base_loss_Q = []
            dual_loss_Q = []
            deep_dual_loss_Q = []
            steps = []
            progress_bar_3 = tqdm(range(repetitions), position=1)
            for i in progress_bar_3:
                X, y = make_classification(n_samples=ns, n_features=nf, class_sep=8,
                                           n_informative=ni2, n_redundant=0, hypercube=True,
                                           n_classes=2, n_clusters_per_class=1, random_state=i)
                X = StandardScaler().fit_transform(X)

                # Base
                inputs = Input(shape=(nf,), name='input')
                model = BaseModel(n_features=nf, k_prototypes=k, deep=False, inputs=inputs, outputs=inputs)
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr_base)
                model.compile(optimizer=optimizer)
                model.layers[1].summary()
                model.fit(X, y, epochs=epochs, verbose=False)
                x_pred = model.predict(X)
                prototypes = model.base_model.weights[-1].numpy()
                accuracy = score(X, prototypes, y)
                print("Accuracy", accuracy, "\n")
                list_acc_base.append(accuracy)
                base_loss_Q.extend(model.loss_)

                # # Deep base
                # inputs = Input(shape=(nf,), name='input')
                # model = BaseModel(n_features=nf, k_prototypes=k, inputs=inputs, outputs=inputs)
                # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_base)
                # model.compile(optimizer=optimizer)
                # model.summary()
                # model.fit(X, y, epochs=epochs)
                # x_pred = model.predict(X)
                # prototypes = model.base_model.weights[-1].numpy()
                # accuracy = score(X, prototypes, y)
                # print("Accuracy", accuracy)
                # list_acc_deep_base.append(accuracy)
                # deep_base_loss_Q.extend(model.loss_)

                # Dual
                inputs = Input(shape=(nf,), name='input')
                model = DualModel(n_samples=ns, k_prototypes=k, deep=False, inputs=inputs, outputs=inputs)
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr_dual)
                model.compile(optimizer=optimizer)
                model.layers[1].summary()
                model.fit(X, y, epochs=epochs, verbose=False)
                x_pred = model.predict(X)
                prototypes = model.dual_model.predict(x_pred.T)
                accuracy = score(X, prototypes, y)
                print("Accuracy", accuracy, "\n")
                list_acc_dual.append(accuracy)
                dual_loss_Q.extend(model.loss_)

                # Deep dual
                inputs = Input(shape=(nf,), name='input')
                model = DualModel(n_samples=ns, k_prototypes=k, deep=True, inputs=inputs, outputs=inputs)
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr_dual)
                model.compile(optimizer=optimizer)
                model.layers[1].summary()
                model.fit(X, y, epochs=epochs, verbose=False)
                x_pred = model.predict(X)
                prototypes = model.dual_model.predict(x_pred.T)
                accuracy = score(X, prototypes, y)
                print("Accuracy", accuracy, "\n")
                list_acc_deep_dual.append(accuracy)
                deep_dual_loss_Q.extend(model.loss_)

                # k-Means
                _, has_samples = compute_graph(x_pred, prototypes, return_has_sampels=True)
                k1 = np.sum(has_samples)
                model_km = KMeans(n_clusters=k1, init='random', random_state=i).fit(X)
                prototypes = model_km.cluster_centers_.T
                loss = quantization(X, prototypes).numpy().astype('float32')
                kmeans_losses.extend(len(model.loss_) * [loss])
                accuracy = score(X, prototypes.astype('float32'), y)
                print("Accuracy", accuracy)
                list_acc_kmeans.append(accuracy)

                steps.extend(np.arange(0, epochs))
                ns_count.append(ns)
                nf_count.append(nf)
                ni_count.append(ni)
                ds_name.append(f'S {ns} - I {ni}')

                # Clearing tf session to cancel previous models
                tf.keras.backend.clear_session()

            losses_Q = pd.DataFrame({
                'epoch': steps,
                'base': base_loss_Q,
                'dual': dual_loss_Q,
                # 'deep-base': deep_base_loss_Q,
                'deep-dual': deep_dual_loss_Q,
                'kmeans': kmeans_losses,
            })

            sns.set_style('whitegrid')
            plt.figure(figsize=[4, 3])
            sns.lineplot('epoch', 'base', data=losses_Q, label='base', ci=99)
            sns.lineplot('epoch', 'dual', data=losses_Q, label='dual', ci=99)
            # sns.lineplot('epoch', 'deep-base', data=losses_Q, label='deep-base', ci=99)
            sns.lineplot('epoch', 'deep-dual', data=losses_Q, label='deep-dual', ci=99)
            sns.lineplot('epoch', 'kmeans', data=losses_Q, label='kmeans', ci=99)
            plt.yscale('log', basey=10)
            plt.ylabel('Q')
            plt.title(f'{dataset}_f_{nf}')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
            plt.tight_layout()

            plt.savefig(os.path.join(results_dir, f'{dataset}_f_{nf}_loss_q.png'))
            plt.savefig(os.path.join(results_dir, f'{dataset}_f_{nf}_loss_q.pdf'))
            plt.show()

            # # Add accuracies of iterations for current dataset
            # list_acc_base += acc_base
            # list_acc_dual += acc_dual
            # list_acc_kmeans += acc_kmeans

        accuracies = pd.DataFrame({
            'number_sample': ns_count,
            'number_feature': nf_count,
            'number_info_feature': ni_count,
            'base': list_acc_base,
            'dual': list_acc_dual,
            # 'deep-base': list_acc_deep_base,
            'deep-dual': list_acc_deep_dual,
            'kmeans': list_acc_kmeans,
        })

        # Saving dataframe to disk in order to recover data if needed
        accuracies.to_pickle(os.path.join(results_dir, f'dataframe_accuracy_#s-{ns}_%i-{ni*100}_#f-{n_features}.png'))

        sns.set_style('whitegrid')
        plt.figure(figsize=[4, 3])
        sns.lineplot('number_feature', 'base', data=accuracies, label='base', ci=99)
        sns.lineplot('number_feature', 'dual', data=accuracies, label='dual', ci=99)
        # sns.lineplot('number_feature', 'deep-base', data=accuracies, label='deep-base', ci=99)
        sns.lineplot('number_feature', 'deep-dual', data=accuracies, label='deep-dual', ci=99)
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
