import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from fexin import Gexin, GHexin, Fexin

X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)

N = 10
model = Gexin(verbose=True)
model.fit(X, N=N, num_epochs=100, lr=0.01)
model.compute_sample_graph()
model.compute_graph()
model.plot_adjacency_matrix()
model.plot_graph(y)
model.plot_sample_graph(y)