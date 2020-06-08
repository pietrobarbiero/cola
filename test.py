import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from fexin import Fexin, FHexin

X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X[:, [0, 2]])
# X = StandardScaler().fit_transform(X)

model = FHexin()
model.fit(X, N_min=10, N_max=11, num_epochs=400, lr=0.01)
model.plot(X, y, "FHexin", "FHexin.png")
