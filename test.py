import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from fexin import Gexin, GHexin, Fexin

X, y = load_iris(return_X_y=True)
X1 = StandardScaler().fit_transform(X[:, [0, 2]])
X2 = StandardScaler().fit_transform(X[:, [1, 3]])
# X = StandardScaler().fit_transform(X)

# model = GHexin()
# # model.fit(X1, N_min=10, N_max=11, num_epochs=400, lr=0.01)
# model.fit(X1, N_min=10, N_max=11, num_epochs=1, lr=0.01)
# model.plot(X1, y, "GHexin", "GHexin.png")

model = Fexin()
model.fit([X1, X2], N_min=10, N_max=11, num_epochs=400, lr=0.01)
model.plot([X1, X2], y, "Fexin", "Fexin.png")
