import os
import unittest

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


class TestDTL(unittest.TestCase):

    def test_class(self):

        from dtl import DeepTopologicalClustering

        X, y = load_iris(return_X_y=True)

        X = StandardScaler().fit_transform(X)

        N = 40
        model = DeepTopologicalClustering()
        model.fit(X, N=N, num_epochs=400, lr=0.0008)
        model.compute_sample_graph()
        model.compute_graph()

        results_dir = "./test-results"
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        model.plot_adjacency_matrix()
        model.plot_graph(y, os.path.join(results_dir, "digits.png"))
        model.plot_sample_graph(y, os.path.join(results_dir, "digits_samples.png"))

        return


suite = unittest.TestLoader().loadTestsFromTestCase(TestDTL)
unittest.TextTestRunner(verbosity=2).run(suite)
