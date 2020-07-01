import os
import unittest

from sklearn.datasets import load_iris, make_blobs
from sklearn.preprocessing import StandardScaler


class TestDTL(unittest.TestCase):

    def test_class(self):

        from deeptl import DeepTopologicalClustering

        X, y = make_blobs(n_samples=200, random_state=42)

        model = DeepTopologicalClustering()
        model.fit(X, N=30, num_epochs=200, lr=0.01)
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
