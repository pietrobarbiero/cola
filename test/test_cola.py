import os
import unittest

from sklearn.datasets import load_iris, make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class TestCOLA(unittest.TestCase):

    def test_class(self):

        from cola import DualModel, plot_confusion_matrix, \
            scatterplot, scatterplot_dynamic, compute_graph, BaseModel

        results_dir = './test-results'
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        dataset = 'circles'
        X, y = make_circles(n_samples=500, factor=.5, noise=.05)
        X = StandardScaler().fit_transform(X)

        n = X.shape[0]
        d = X.shape[1]
        k = 40
        lr_vanilla = 0.008
        lr_dual = 0.00008
        epochs = 800

        inputs = Input(shape=(d,), name='input')
        outputs = inputs
        model = BaseModel(n_features=d, k_prototypes=k, inputs=inputs, outputs=outputs, deep=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_vanilla)
        model.compile(optimizer=optimizer)
        model.summary()
        model.fit(X, y, epochs=epochs, verbose=True)
        x_pred = model.predict(X)
        prototypes = model.base_model.weights[-1].numpy()
        G = compute_graph(x_pred, prototypes)
        plt.figure()
        plot_confusion_matrix(x_pred, prototypes, y)
        plt.savefig(os.path.join(results_dir, f'{dataset}_confmat_vanilla.png'))
        plt.show()
        plt.figure()
        scatterplot(x_pred, prototypes, y, valid=True)
        plt.savefig(os.path.join(results_dir, f'{dataset}_scatter_vanilla.png'))
        plt.show()
        plt.figure()
        scatterplot_dynamic(X, model.prototypes_, y, valid=True)
        plt.savefig(os.path.join(results_dir, f'{dataset}_dynamic_vanilla.png'))
        plt.show()

        inputs = Input(shape=(d,), name='input')
        model = DualModel(n_samples=n, k_prototypes=k, inputs=inputs, outputs=inputs, deep=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_dual)
        model.compile(optimizer=optimizer)
        model.summary()
        model.fit(X, y, epochs=epochs)
        x_pred = model.predict(X)
        prototypes = model.dual_model.predict(x_pred.T)
        G = compute_graph(x_pred, prototypes)
        plt.figure()
        plot_confusion_matrix(x_pred, prototypes, y)
        plt.savefig(os.path.join(results_dir, f'{dataset}_confmat_dual.png'))
        plt.show()
        plt.figure()
        scatterplot(x_pred, prototypes, y, valid=True)
        plt.savefig(os.path.join(results_dir, f'{dataset}_scatter_dual.png'))
        plt.show()
        plt.figure()
        scatterplot_dynamic(X, model.prototypes_, y, valid=True)
        plt.savefig(os.path.join(results_dir, f'{dataset}_dynamic_dual.png'))
        plt.show()

        return


suite = unittest.TestLoader().loadTestsFromTestCase(TestCOLA)
unittest.TextTestRunner(verbosity=2).run(suite)
