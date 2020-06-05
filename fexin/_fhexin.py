from tqdm import tqdm
import numpy as np
import tensorflow as tf

from ._fexin import Fexin


class FHexin():
    def __init__(self, optimizer=None, verbose=True):
        self.optimizer = optimizer
        self.verbose = verbose

    def fit(self, X, num_epochs=200):
        self.best_loss_ = np.inf
        self.fexin_ = None
        pbar = tqdm(range(2, X.shape[0]))
        for level in pbar:

            input = tf.keras.layers.Input(shape=(X.shape[0],))
            x = tf.keras.layers.Dense(10, activation='relu')(input)
            x = tf.keras.layers.Dense(10, activation='relu')(x)
            output = tf.keras.layers.Dense(level)(x)
            kmodel = tf.keras.Model(inputs=input, outputs=output)

            fexin = Fexin(kmodel, optimizer=self.optimizer, verbose=False)
            fexin.fit(X, N=level, num_epochs=num_epochs)
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
