import tensorflow as tf
from tensorflow.keras import layers, Model, metrics
from tqdm import tqdm

from ._loss import quantization, silhouette

mae_metric = metrics.MeanAbsoluteError(name="mae")
loss_tracker = metrics.Mean(name="loss")


class DualModel(Model):

    def __init__(self, n_samples, k_prototypes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input = layers.Input(shape=(n_samples,))
        # x = layers.Dense(k_prototypes, activation='tanh')(input)
        # x = layers.Dense(k_prototypes, activation='tanh')(x)
        # output = layers.Dense(k_prototypes)(x)
        output = layers.Dense(k_prototypes)(input)
        self.dual_model = tf.keras.Model(inputs=input, outputs=output)

    def fit(self, X, y, epochs):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        pbar = tqdm(range(epochs))
        x = tf.Variable(X, dtype='float32')
        for epoch in pbar:
            with tf.GradientTape() as tape:
                y_latent = self(x, training=False)  # Forward pass
                y_pred = self.dual_model(tf.transpose(y_latent), training=True)
                loss = quantization(y_latent, y_pred)

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            # Compute our own metrics
            loss_tracker.update_state(loss)

            pbar.set_description(f"Epoch: {epoch+1} - Loss: {loss.numpy():.2f}")
        return self
