import tensorflow as tf
from tensorflow.keras import layers, Model, metrics
from tqdm import tqdm

from ._loss import quantization

mae_metric = metrics.MeanAbsoluteError(name="mae")
loss_tracker = metrics.Mean(name="loss")


class BaseModel(Model):

    def __init__(self, n_features, k_prototypes, deep=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input = layers.Input(shape=(n_features,))
        if deep:
            x = layers.Dense(n_features, activation='tanh')(input)
            x = layers.Dense(n_features, activation='tanh')(x)
            output = layers.Dense(k_prototypes, use_bias=False)(x)
        else:
            output = layers.Dense(k_prototypes, use_bias=False)(input)
        self.base_model = tf.keras.Model(inputs=input, outputs=output)

    def fit(self, X, y, epochs, verbose=False):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        pbar = tqdm(range(epochs)) if verbose else None
        x = tf.Variable(X, dtype='float32')
        self.loss_ = []
        self.prototypes_ = []
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                y_latent = self(x, training=True)  # Forward pass
                loss = quantization(y_latent, self.base_model.weights[-1])

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            # Compute our own metrics
            loss_tracker.update_state(loss)
            self.loss_.append(loss.numpy())
            self.prototypes_.append(self.base_model.weights[-1].numpy())

            if verbose:
                pbar.update(1)
                pbar.set_description(f"Epoch: {epoch+1} - Loss: {loss.numpy():.2f}")
        return self
