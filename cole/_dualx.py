import tensorflow as tf
from tensorflow.keras import layers, Model, metrics
from tqdm import tqdm

from ._loss import quantization, silhouette

mae_metric = metrics.MeanAbsoluteError(name="mae")
loss_tracker = metrics.Mean(name="loss")


class DualXModel(Model):

    def __init__(self, n_samples, k_prototypes, deep=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input = layers.Input(shape=(n_samples,))
        if deep:
            x = layers.Dense(k_prototypes, activation='tanh')(input)
            x = layers.Dense(k_prototypes, activation='tanh')(x)
            output = layers.Dense(k_prototypes)(x)
        else:
            x = layers.BatchNormalization()(input)
            output = layers.Dense(k_prototypes, use_bias=False)(x)
        self.dual_model = tf.keras.Model(inputs=input, outputs=output)

    def fit(self, X, y, epochs, verbose=True):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        pbar = tqdm(range(epochs)) if verbose else None
        x = tf.Variable(X, dtype='float32')
        self.loss_ = []
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                output = tf.reduce_min(self.dual_model.weights[-1], axis=1)
                loss = tf.norm(output)

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            # Compute our own metrics
            loss_tracker.update_state(loss)
            self.loss_.append(loss.numpy())
            if verbose:
                pbar.set_description(f"Epoch: {epoch+1} - Loss: {loss.numpy():.2f}")
        return self
