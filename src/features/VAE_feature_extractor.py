import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

dim = 128

def encoder(latent_dim=dim, input_shape=(64, 64, 3)):
    '''
    
    Define the encoder architecture
    
    '''
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    
    return encoder


def decoder(latent_dim=dim):
    '''
    
    Define the encoder architecture
    
    '''
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    
    return decoder



class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    

def given_a_list_of_images_return_a_batch(list_of_image_patches):
    '''
    Return a batched array when given a list of images
    
    '''
    image_patches = [np.expand_dims(patch, axis=0) for patch in list_of_image_patches]

    batch_of_all_images = np.concatenate(image_patches, axis=0).astype(np.float32) / 255
    
    return batch_of_all_images
    
    
def train_VAE_model(list_of_image_patches, path_to_save_trained_model, epochs=30, batch_size=128):
    '''
    Train VAE to learn a feature extractor from the image patches
    
    '''
    
    batch_of_all_images = given_a_list_of_images_return_a_batch(list_of_image_patches)
    
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(batch_of_all_images, epochs=epochs, batch_size=batch_size)
    
    vae.save(path_to_save_trained_model)
    

def extract_features_using_VAE(list_of_image_patches, path_to_trained_model, train_model=True):
    '''
    Given a list of image patches, use VAE to extract their features
    
    '''
    if train_model:
        
        train_VAE_model(list_of_image_patches, path_to_trained_model, epochs=30, batch_size=128)
        
    model = keras.models.load_model(path_to_trained_model)
    
    batch_of_all_images = given_a_list_of_images_return_a_batch(list_of_image_patches)
    
    z_mean, z_log_var, z = vae.encoder.predict(data)
    
    matrix_of_feature_vectors = z_mean
    
    print(f'Extracted VAE data matrix of shape: {matrix_of_feature_vectors.shape}')
    
    return matrix_of_feature_vectors
    
    


