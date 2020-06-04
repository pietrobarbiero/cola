import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from fexin import Fexin, FHexin

X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X[:, [0, 2]])

model = FHexin()
model.fit(X, num_epochs=200)
model.plot(X, "FHexin", "FHexin.png")
