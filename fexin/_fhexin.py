from tqdm import tqdm
import numpy as np
import tensorflow as tf

from ._fexin import Fexin


class FHexin():
    def __init__(self, optimizer=None, verbose=True):
        self.optimizer = optimizer
        self.verbose = verbose

    def fit(self, X, N_min=None, N_max=None, num_epochs=200, lr=0.01, beta_o=0.1):
        self.best_loss_ = np.inf
        self.fexin_ = None

        if N_min is None:
            N_min = 2

        if N_max is None:
            N_max = X.shape[0]

        if self.optimizer is None:
            self.optimizer_ = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            self.optimizer_ = self.optimizer

        pbar = tqdm(range(N_min, N_max))
        for level in pbar:

            y_idx = np.random.choice(range(X.shape[0]), size=level, replace=False)
            y = X[y_idx]
            input = tf.keras.layers.Input(shape=(X.shape[0],))
            x = tf.keras.layers.Dense(10, activation='relu')(input)
            x = tf.keras.layers.Dense(10, activation='relu')(x)
            output = tf.keras.layers.Dense(level)(x)
            kmodel = tf.keras.Model(inputs=input, outputs=output)
            kmodel.compile(optimizer=self.optimizer_, loss="mse")
            kmodel.fit(X.T, y.T, epochs=100, verbose=0)

            fexin = Fexin(kmodel, optimizer=self.optimizer_, verbose=False)
            fexin.fit(X, N=level, num_epochs=num_epochs, beta_o=beta_o)
            loss = fexin.loss_value_

            pbar.set_description(f"Level: {level} - Loss: {loss:.2f}")

            if loss < self.best_loss_:
                self.best_loss_ = loss
                self.fexin_ = fexin
            else:
                break

        return self

    def predict(self, X):
        return self.fexin_.predict(X)

    def plot(self, X, y=None, title="", file_path="fhexin.png"):
        return self.fexin_.plot(X, y, title, file_path)
