# import sys
# sys.path.append('./')

# from models.core_utils import segment_image_and_extract_segment_features
# from models.predict_model import save_patches_to_directory, generate_csv_summarizing_detections
# import glob
# import imageio
# import matplotlib.pyplot as plt
# import numpy as np
# import PIL
# import tensorflow as tf
# import numpy as np
# #import tensorflow_probability as tfp
# import time
# from pathlib import Path
# from sklearn.decomposition import PCA
# from numpy.random import default_rng 
# import pandas as pd
# import seaborn as sns
# from skimage.io import imread
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# from sklearn.decomposition import KernelPCA
# from sklearn.neighbors import NeighborhoodComponentsAnalysis
# from sklearn.covariance import EllipticEnvelope
# from sklearn import svm
# from sklearn.ensemble import IsolationForest
# from math import ceil
# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# import random
# from functools import partial
# from models.core_utils import merge_segmentation_patches_from_all_images
# import concurrent.futures
# from joblib import Parallel, delayed
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from itertools import compress
# import shutil
# import time
# from keras import backend as K

# rng = default_rng()


# def create_tensorflow_dataset_from_numpy_ndarray(list_of_ndarrays, batch_size, is_for_training=None):
#     '''
#     Convert numpy ndarray to batched tensor slices
    
#     '''
#     batch_of_ndarrays = np.stack(list_of_ndarrays).astype('uint8')
    
#     data_gen_for_training = ImageDataGenerator(validation_split=0.2)
    
#     data_gen_for_testing = ImageDataGenerator()
    
#     total_number_of_batches = ceil(len(batch_of_ndarrays)/batch_size)
    
#     number_of_train_batches = ceil(0.8 * total_number_of_batches)
    
#     number_of_val_batches = ceil(0.2 * total_number_of_batches)
    
#     if is_for_training:
                
#         train_generator = data_gen_for_training.flow(batch_of_ndarrays, batch_size=batch_size, y=None, subset='training')
        
#         val_generator = data_gen_for_training.flow(batch_of_ndarrays, batch_size=batch_size, y=None, subset='validation')
                
#         return train_generator, val_generator, number_of_train_batches, number_of_val_batches
        
#     else:
        
#         test_generator = data_gen_for_testing.flow(batch_of_ndarrays, batch_size=batch_size, shuffle=False, y=None)

#         return test_generator, total_number_of_batches
       
        
# def read_images_from_disk(directory_containing_images, batch_size=32, im_height=96, im_width=96):
#     '''
#     Read images rom disk and split to train test
    
#     '''
#     directory_containing_images = Path(directory_containing_images)
    
#     train_ds = tf.keras.utils.image_dataset_from_directory(
#     directory_containing_images,
#     validation_split=0.2,
#     subset='training',
#     seed=123,
#     image_size=(im_height, im_width),
#     batch_size=batch_size,
#     label_mode=None)
    
#     val_ds = tf.keras.utils.image_dataset_from_directory(
#     directory_containing_images,
#     validation_split=0.2,
#     subset='validation',
#     seed=123,
#     image_size=(im_height, im_width),
#     batch_size=batch_size,
#     label_mode=None)
    
#     return train_ds, val_ds

# ################## END OF DATA LOADING UTILS ########################################



# ############################ VAE MODEL DEFINITION #################################
# n_filters = 12

# class CVAE(tf.keras.Model):
    
#   """Convolutional variational autoencoder."""

#   def __init__(self, latent_dim):
#     super(CVAE, self).__init__()
#     self.latent_dim = latent_dim
#     self.encoder = tf.keras.Sequential(
#         [
#              tf.keras.layers.InputLayer(input_shape=(96, 96, 3)),
            
#             tf.keras.layers.Conv2D(
#                 filters=1*n_filters, kernel_size=3, strides=(2, 2),  padding='same', activation='relu'),
#             tf.keras.layers.BatchNormalization(),
            
#             tf.keras.layers.Conv2D(
#                 filters=2*n_filters, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
#             tf.keras.layers.BatchNormalization(),
            
#             tf.keras.layers.Conv2D(
#                 filters=4*n_filters, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
#             tf.keras.layers.BatchNormalization(),
            
#             tf.keras.layers.Conv2D(
#                 filters=6*n_filters, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
#             tf.keras.layers.BatchNormalization(),
            
#             tf.keras.layers.Flatten(),
#             #tf.keras.layers.Dense(512),
#             # No activation
#             tf.keras.layers.Dense(latent_dim + latent_dim),
#         ]
#     )

#     self.decoder = tf.keras.Sequential(
#         [
#             tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            
#             tf.keras.layers.Dense(2592, activation='relu'),
    
#             tf.keras.layers.Reshape(target_shape=(6, 6, 72)),

#             # Upscaling convolutions (inverse of encoder)
#             tf.keras.layers.Conv2DTranspose(filters=4*n_filters, kernel_size=3,  strides=2, padding='same', activation='relu'),
#             tf.keras.layers.Conv2DTranspose(filters=2*n_filters, kernel_size=3,  strides=2, padding='same', activation='relu'),
#             tf.keras.layers.Conv2DTranspose(filters=1*n_filters, kernel_size=3,  strides=2, padding='same', activation='relu'),
#             tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3,  strides=2, padding='same'),
#           ]
#     )
    
    
    

#   @tf.function
#   def sample(self, eps=None):
        
#         if eps is None:
            
#             eps = tf.random.normal(shape=(100, self.latent_dim))
            
#         return self.decode(eps, apply_sigmoid=True)

#   def encode(self, x):
                
#         mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        
#         return mean, logvar
    
    
#   def reparameterize(self, mean, logvar):
    
    
#     eps = tf.random.normal(shape=mean.shape)
    
#     return eps * tf.exp(logvar * .5) + mean

#   def decode(self, z, apply_sigmoid=False):
        
#         logits = self.decoder(z)
        
#         if apply_sigmoid:
            
#             probs = tf.sigmoid(logits)
            
#             return probs
        
#         return logits
    
    
#   def call(self, inputs):

#         vector = self.encoder(inputs)

# ############################ END OF VAE MODEL DEFINITION #################################


# ################################# VAE UTILS #############################################
# def log_normal_pdf(sample, mean, logvar, raxis=1):
    
#     log2pi = tf.math.log(2. * np.pi)
    
#     return tf.reduce_sum(
#       -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
#       axis=raxis)


# def compute_loss(model, x):
    
#     mean, logvar = model.encode(x)
    
#     z = model.reparameterize(mean, logvar)
    
#     x_logit = model.decode(z)
    
# #     cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    
# #     logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    
#     mse = tf.keras.losses.MeanSquaredError()
    
#     logpx_z = -mse(x, x_logit)
    
#     logpz = log_normal_pdf(z, 0., 0.)
    
#     logqz_x = log_normal_pdf(z, mean, logvar)
    
#     logqz_x = log_normal_pdf(z, mean, logvar)
    
#     return -tf.reduce_mean(logpx_z + (logpz - logqz_x))




# @tf.function
# def train_step(model, x, optimizer):
    
#     """
#     Executes one training step and returns the loss.

#     This function computes the loss and gradients, and uses the latter to
#     update the model's parameters.
#     """
#     with tf.GradientTape() as tape:
        
#         loss = compute_loss(model, x)
        
#         gradients = tape.gradient(loss, model.trainable_variables)
        
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
# ################################# END OF VAE UTILS #############################################


# ################################## VAE MODEL TRAINING ##########################################
# def train_VAE(latent_dim, train_generator, val_generator, number_of_train_batches, number_of_val_batches, epochs = 10):
#     '''
#     Train the VAE
    
#     ''' 
#     all_models = []
#     all_ELBO_losses = []
    
#     optimizer = tf.keras.optimizers.Adam(1e-4)

#     model = CVAE(latent_dim)

#     for epoch in range(1, epochs + 1):
        
#         start_time = time.time()
        
#         for train_x, _ in zip(train_generator, range(number_of_train_batches)):
            
#             train_step(model, train_x/255, optimizer)
            
#         end_time = time.time()

#         loss = tf.keras.metrics.Mean()
        
#         for test_x, _ in zip(val_generator,range(number_of_val_batches)):
            
#             loss(compute_loss(model, test_x/255))
            
#         elbo = -loss.result()
        
#         all_models.append(model)
#         all_ELBO_losses.append(elbo)
#       #display.clear_output(wait=False)
#         print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {:.2f} seconds'
#             .format(epoch, elbo, end_time - start_time))
        
    
#     best_model = retrieve_the_best_performing_model(all_models, all_ELBO_losses)
    
#     input_arr = tf.random.uniform((2,96,96,3))
    
#     encoder = model.encoder
    
#     mean_vector, logvar = encoder(input_arr)
    
#     return encoder

# def retrieve_the_best_performing_model(all_models, all_ELBO_losses):
#     '''
#     Choose the model that performs the best of all epochs
    
#     '''
#     index_of_lowest_loss = np.argmin(all_ELBO_losses)
    
#     best_model = all_models[index_of_lowest_loss]
    
#     print(f'The VAE model with lowest ELBO was at epoch {index_of_lowest_loss + 1} of {len(all_ELBO_losses)}')
    
#     return best_model

# ################################## END OF VAE MODEL TRAINING ##########################################



# ######################################## CORE UTILS ##########################################

# def segment_images_and_return_segments_as_list_of_ndarrays(list_of_file_paths):
#     '''
#     Given a directory of images, segment them and return the superpixels as ndarrays
    
#     '''
#     all_file_paths = list_of_file_paths #list(directory_containing_images.rglob('*.JPG'))
    
#     random.shuffle(all_file_paths)
    
#     #all_file_paths = random.sample(all_file_paths, k=10)
    
#     segmented_image_objects = []
    
#     try:
        
#         with ProcessPoolExecutor(14) as executor:

#             _segment_image_and_extract_segment_features = partial(segment_image_and_extract_segment_features, training_mode=False) #Edit to remove train_mode flag

#             segmented_image_objects = list(executor.map(_segment_image_and_extract_segment_features, all_file_paths))
                
            
#     except Exception as excp:
#         print(f'Cannot use multithreading. Error message: {excp}')
        
#         _segment_image_and_extract_segment_features = partial(segment_image_and_extract_segment_features, training_mode=False) #Edit to remove train_mode flag

#         # segmented_image_objects = list(map(_segment_image_and_extract_segment_features, all_file_paths))
        
#         segmented_image_objects = Parallel(n_jobs=14)(delayed(_segment_image_and_extract_segment_features)(fp) for fp in all_file_paths)
        
        
#     #Gather segment patches and support sets
#     segment_patches, segment_patch_names, segment_patch_bboxes, segment_patch_class_labels = merge_segmentation_patches_from_all_images(segmented_image_objects)
    
#     segmented_images = [segmented_image_object.segmented_image for segmented_image_object in segmented_image_objects][:10]
    
#     original_images = [segmented_image_object.rgb_image for segmented_image_object in segmented_image_objects][:10]
        
#     return segment_patches, segment_patch_names, segment_patch_bboxes, segment_patch_class_labels, segmented_images, original_images



# def extract_features_using_trained_VAE(trained_VAE_encoder_model, data_generator, number_of_batches):
#     '''
#     Use a trained VAE encoder to extract feature vectors
    
#     '''
#     training_features = []
    
#     for batch, _ in zip(data_generator, range(number_of_batches)):
        
#         mean_vector, logvar = tf.split(trained_VAE_encoder_model(batch/255), num_or_size_splits=2, axis=1)
        
#         training_features.append(mean_vector)
    
#     training_features = np.vstack(training_features)
    
#     return training_features


# def copy_training_images_from_parent_images(directory_containing_test_images, directory_containing_training_images, sample_size):
#     '''
#     Sample a number of images to be used to train the VAE
    
#     '''
#     directory_containing_training_images = Path(directory_containing_training_images)
    
#     directory_containing_test_images = Path(directory_containing_test_images)
    
#     file_paths = list(directory_containing_test_images.rglob('*.JPG'))
    
#     random.shuffle(file_paths)
    
#     sampled_file_paths = random.sample(file_paths, k=sample_size)
    
#     #Delete previously sampled training images if any
#     shutil.rmtree(directory_containing_training_images, ignore_errors=True)
#     directory_containing_training_images.mkdir()
    
#     with ThreadPoolExecutor() as executor:
        
#         [executor.submit(shutil.copy2, file_path, directory_containing_training_images) for file_path in sampled_file_paths]
        
#     return

####################### RE-ENGINEERED ####################

import sys
sys.path.append('./')

from models.core_utils import segment_image_and_extract_segment_features
from models.predict_model import save_patches_to_directory, generate_csv_summarizing_detections
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import numpy as np
#import tensorflow_probability as tfp
import time
from pathlib import Path
from sklearn.decomposition import PCA
from numpy.random import default_rng 
import pandas as pd
import seaborn as sns
from skimage.io import imread
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.covariance import EllipticEnvelope
from sklearn import svm
from sklearn.ensemble import IsolationForest
from math import ceil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import random
from functools import partial
from models.core_utils import merge_segmentation_patches_from_all_images
import concurrent.futures
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from itertools import compress
import shutil
import time
from keras import backend as K

from parameters import deepsea_fauna_detection_params
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model
from skimage.transform import resize
from itertools import compress

rng = default_rng()

################## DATA LOADING UTILS ########################################
def create_tensorflow_dataset_from_numpy_ndarray(list_of_ndarrays, batch_size, is_for_training=None):
    '''
    Convert numpy ndarray to batched tensor slices
    
    '''
    batch_of_ndarrays = np.stack(list_of_ndarrays).astype('uint8')
    
    data_gen_for_training = ImageDataGenerator(validation_split=0.2, rescale=1/255.)
    
    data_gen_for_testing = ImageDataGenerator(rescale=1/255.)
    
    total_number_of_batches = ceil(len(batch_of_ndarrays)/batch_size)
    
    number_of_train_batches = ceil(0.8 * total_number_of_batches)
    
    number_of_val_batches = ceil(0.2 * total_number_of_batches)
    
    if is_for_training:
                
        train_generator = data_gen_for_training.flow(batch_of_ndarrays, batch_size=batch_size, y=None, subset='training')
        
        val_generator = data_gen_for_training.flow(batch_of_ndarrays, batch_size=batch_size, y=None, subset='validation')
                
        return train_generator, val_generator, number_of_train_batches, number_of_val_batches
        
    else:
        
        test_generator = data_gen_for_testing.flow(batch_of_ndarrays, batch_size=batch_size, shuffle=False, y=None)

        return test_generator, total_number_of_batches
################## END OF DATA LOADING UTILS ########################################

    
############################ VAE MODEL DEFINITION #################################    
n_filters = 12

class CVAE(tf.keras.Model):
    
    """
    Convolutional variational autoencoder.

    """
    

    def __init__(self, latent_dim):
        
        super(CVAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.encoder = tf.keras.Sequential(
            
            [
                 tf.keras.layers.InputLayer(input_shape=(96, 96, 3)),

                tf.keras.layers.Conv2D(
                    filters=1*n_filters, kernel_size=3, strides=(2, 2),  padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Conv2D(
                    filters=2*n_filters, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Conv2D(
                    filters=4*n_filters, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Conv2D(
                    filters=6*n_filters, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),

                tf.keras.layers.Dense(2592, activation='relu'),

                tf.keras.layers.Reshape(target_shape=(6, 6, 72)),

                # Upscaling convolutions (inverse of encoder)
                tf.keras.layers.Conv2DTranspose(filters=4*n_filters, kernel_size=3,  strides=2, padding='same', activation='relu'),
                #tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2DTranspose(filters=2*n_filters, kernel_size=3,  strides=2, padding='same', activation='relu'),
                #tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2DTranspose(filters=1*n_filters, kernel_size=3,  strides=2, padding='same', activation='relu'),
                #tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3,  strides=2, padding='same'),
              ]
        )
    
    @tf.function
    def sample(self, eps=None):

        if eps is None:

            eps = tf.random.normal(shape=(100, self.latent_dim))

        return self.decode(eps)

    def encode(self, x):

        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)

        return mean, logvar


    def reparameterize(self, mean, logvar):

        eps = tf.random.normal(shape=mean.shape)

        return eps * tf.exp(logvar * .5) + mean


    def decode(self, z):

        logits = self.decoder(z)

        return logits


    def call(self, inputs):

        vector = self.encoder(inputs)
        
        return vector
    
 ############################ END OF VAE MODEL DEFINITION ################################# 

################################# VAE UTILS #############################################
def log_normal_pdf(sample, mean, logvar, raxis=1):
    
    log2pi = tf.math.log(2. * np.pi)
    
    return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)


# def compute_loss(model, x):
    
#     mean, logvar = model.encode(x)
    
#     z = model.reparameterize(mean, logvar)
    
#     x_logit = model.decode(z)
    
#     mse = tf.keras.losses.MeanSquaredError()
    
#     #logpx_z = -mse(x, x_logit)
    
#     logpx_z = tf.math.log(tf.math.sigmoid(mse(x, x_logit)))
    
#     logpz = log_normal_pdf(z, 0., 0.) #log of 1(std) is zero
    
#     logqz_x = log_normal_pdf(z, mean, logvar)
    
#     return -tf.reduce_mean(logpx_z + 0.0005*(logpz - logqz_x))

def compute_loss(model, x):
    
    mu, logsigma = model.encode(x)
    
    z = model.reparameterize(mu, logsigma)
    
    x_recon = model.decode(z)
    
    kl_weight = 0.5

    latent_loss = 0.5 * tf.reduce_sum(tf.exp(logsigma) + tf.square(mu) - 1.0 - logsigma, axis=1)

    reconstruction_loss = tf.reduce_mean(tf.abs(x-x_recon), axis=(1,2,3))

    vae_loss = kl_weight * latent_loss + reconstruction_loss

    return vae_loss

@tf.function
def train_step(model, x, optimizer):
    
    """
    Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        
        loss = compute_loss(model, x)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
    return 
        
################################# END OF VAE UTILS #############################################


################################## VAE MODEL TRAINING ##########################################
def train_VAE(latent_dim, train_generator, val_generator, number_of_train_batches, number_of_val_batches,directory_containing_pickled_items, epochs = 10):
    '''
    Train the VAE
    
    ''' 
    all_models = []
    all_ELBO_losses = []
    
    optimizer = tf.keras.optimizers.Adam(1e-4)

    model = CVAE(latent_dim)

    for epoch in range(1, epochs + 1):
        
        start_time = time.time()
        
        for train_x, _ in zip(train_generator, range(number_of_train_batches)):
            
            train_step(model, train_x, optimizer)
            
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        
        for test_x, _ in zip(val_generator,range(number_of_val_batches)):
            
            loss(compute_loss(model, test_x))
            
        elbo = -loss.result()
        
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {:.2f} seconds'
            .format(epoch, elbo, end_time - start_time))
        
        input_arr = tf.random.uniform((1,96,96,3))
        
        #encoder = model.encoder

        #mean_vector, logvar = encoder(input_arr)
        
        _ = model(input_arr)
        
        
        if epoch == 1:
            
            all_ELBO_losses.append(elbo)
            
            print('Pickling original VAE model ...')

            model_file_path = str(directory_containing_pickled_items / 'trained_VAE_model')

            model.save(model_file_path)
            
        
        elif (epoch > 1) and (elbo > max(all_ELBO_losses)):
            
            all_ELBO_losses.append(elbo)
            
            print(f'Pickling VAE model for epoch {epoch} ...')
                
            model_file_path = str(directory_containing_pickled_items / 'trained_VAE_model')
    
            model.save(model_file_path)

    
    return

################################## END OF VAE MODEL TRAINING ##########################################


######################################## CORE UTILS ##########################################

def segment_images_and_return_segments_as_list_of_ndarrays(list_of_file_paths):
    '''
    Given a directory of images, segment them and return the superpixels as ndarrays
    
    '''
    all_file_paths = list_of_file_paths
    
    segmented_image_objects = []
    
    try:
        
        with ProcessPoolExecutor(14) as executor:

            segmented_image_objects = list(executor.map(segment_image_and_extract_segment_features, all_file_paths))
                
            
    except Exception as excp:
        
        print(f'Cannot use multithreading. Error message: {excp}')
        
        segmented_image_objects = Parallel(n_jobs=14)(delayed(segment_image_and_extract_segment_features)(fp) for fp in all_file_paths)
        
        
    #Gather segment patches and support sets
    segment_patches, segment_patch_names, segment_patch_bboxes, segment_patch_class_labels = merge_segmentation_patches_from_all_images(segmented_image_objects)
    
    segmented_images = [segmented_image_object.segmented_image for segmented_image_object in segmented_image_objects][:10]
    
    original_images = [segmented_image_object.rgb_image for segmented_image_object in segmented_image_objects][:10]
        
    return segment_patches, segment_patch_names, segment_patch_bboxes, segment_patch_class_labels, segmented_images, original_images



def extract_features_using_trained_VAE(trained_VAE_encoder_model, data_generator, number_of_batches):
    '''
    Use a trained VAE encoder to extract feature vectors
    
    '''
    training_features = []
    
    for batch, _ in zip(data_generator, range(number_of_batches)):
        
        mean_vector, logvar = tf.split(trained_VAE_encoder_model(batch), num_or_size_splits=2, axis=1)
        
        training_features.append(mean_vector)
    
    training_features = np.vstack(training_features)
    
    return training_features


def copy_training_images_from_parent_images(directory_containing_test_images, directory_containing_training_images, sample_size):
    '''
    Sample a number of images to be used to train the VAE
    
    '''
    directory_containing_training_images = Path(directory_containing_training_images)
    
    directory_containing_test_images = Path(directory_containing_test_images)
    
    file_paths = list(directory_containing_test_images.rglob('*.JPG'))
    
    random.shuffle(file_paths)
    
    sampled_file_paths = random.sample(file_paths, k=sample_size)
    
    #Delete previously sampled training images if any
    shutil.rmtree(directory_containing_training_images, ignore_errors=True)
    directory_containing_training_images.mkdir()
    
    with ThreadPoolExecutor() as executor:
        
        [executor.submit(shutil.copy2, file_path, directory_containing_training_images) for file_path in sampled_file_paths]
        
    return

################# END OF RE-ENGINEERING ##################
        

########################### OUTLIER DETECTION USING TRAINED VAE  ######################
    
def train_Isolation_Forest_to_detect_anomalous_patches(background_features, contamination='auto'):
    '''
    Train an instance of IF to be used as outlier detector
    
    '''
    #Train outlier detector    
    outlier_detector = IsolationForest(n_jobs=14, 
                                       bootstrap=True,
                                       max_samples=10_000,
                                       max_features=0.8,
                                       warm_start=False,
                                       contamination=contamination).fit(background_features)
    
    return outlier_detector


def detect_anomalous_patches_using_trained_Isolation_Forest(trained_VAE,
                                      trained_Isolation_Forest, 
                                      directory_containing_test_images,
                                      directory_to_save_patches_of_positive_detections):
    '''
    Use trained VAE to make predictions
    
    '''
    test_outliers_features = []

    test_outliers_patch_names = [] 
    
    test_outliers_patch_bboxes = []

    test_outlier_scores = []
  

    #Split the training set so we can process the in batches
    list_of_file_paths_for_test_images = list(directory_containing_test_images.rglob('*.JPG'))
    
    number_of_splits = ceil(len(list_of_file_paths_for_test_images) / 16)
    
    file_path_partitions = np.array_split(list_of_file_paths_for_test_images, number_of_splits)
    
    #Loop through each batch for memory efficiency
    for file_path_partition in file_path_partitions:
    
        print('Segmenting training images ...')
        list_of_test_patches, test_patch_names, test_patch_bboxes, _, _, _ = segment_images_and_return_segments_as_list_of_ndarrays(file_path_partition.tolist())

        test_generator, number_of_test_batches = create_tensorflow_dataset_from_numpy_ndarray(list_of_test_patches, batch_size=32, is_for_training=False)
        
        test_features_for_this_partition = extract_features_using_trained_VAE(trained_VAE, test_generator, number_of_test_batches)

        #trained_Isolation_Forest = train_Isolation_Forest_to_detect_anomalous_patches(test_features_for_this_partition, contamination='auto')
        
        
        #Detect outliers from the test set partition
        test_outlier_predictions_for_this_partition = trained_Isolation_Forest.predict(test_features_for_this_partition)

        test_outlier_selector_for_this_partition = (test_outlier_predictions_for_this_partition == -1)
        
        #test_scores_for_this_partition = trained_Isolation_Forest.score_samples(test_features_for_this_partition)
        
        test_scores_for_this_partition = trained_Isolation_Forest.decision_function(test_features_for_this_partition)

        test_outliers_features_for_this_partition = np.compress(test_outlier_selector_for_this_partition, test_features_for_this_partition, axis=0)
        
        test_outlier_scores_for_this_partition = np.compress(test_outlier_selector_for_this_partition, test_scores_for_this_partition)

        test_outliers_patches_for_this_partition = list(compress(list_of_test_patches,test_outlier_selector_for_this_partition))
        
        test_patch_names_for_this_partition = list(compress(test_patch_names,test_outlier_selector_for_this_partition))
        
        test_patch_bboxes_for_this_partition = list(compress(test_patch_bboxes,test_outlier_selector_for_this_partition))
                
        test_outliers_features.append(test_outliers_features_for_this_partition)
                
        test_outliers_patch_names.extend(test_patch_names_for_this_partition)
        
        test_outliers_patch_bboxes.extend(test_patch_bboxes_for_this_partition)
                
        test_outlier_scores.extend(test_outlier_scores_for_this_partition.tolist())
        
        #Save the patches as you go
        save_patches_to_directory(directory_to_save_patches_of_positive_detections, test_outliers_patches_for_this_partition, test_patch_names_for_this_partition)
        
        #Clear gpu memory
        K.clear_session()
        
        
    test_outliers_features = np.concatenate(test_outliers_features, axis=0)
    
    n_outliers = len(test_outliers_features)

    print(f'There are {n_outliers} anomalies detected ...')
    
    
    if n_outliers == 0:
        
        return
    
    else:

        pca_object = PCA(n_components=2, whiten=True).fit(test_outliers_features)

        pca_embeddings = pca_object.transform(test_outliers_features)


    print(f'Found {len(pca_embeddings)} outliers ...')

    generate_csv_summarizing_detections(test_outliers_patch_names, test_outliers_features, test_outliers_patch_bboxes, pca_embeddings, test_outlier_scores, directory_to_save_patches_of_positive_detections)

    return

########################## END OF OUTLIER DETECTION USING TRAINED VAE #####################################


########################## VISUALIZATION OF EMBEDDED PATCHES ########################################
def visualize_embedded_segment_patches(embeddings, patches, figsize=(20,12),points_only=True, figname = None, directory_to_save_matplotlib_figures=None, zoom=0.3):
    '''
    Plot the embedding in 2D feature space
    
    '''      
    assert embeddings.shape[1] == 2, 'Project your data matrix to two dimensions'

    df = pd.DataFrame({'pca_0':embeddings[:,0], 'pca_1':embeddings[:,1]})
    
    fig, ax = plt.subplots(figsize=figsize)
    
    temp = sns.scatterplot(x='pca_0', y='pca_1', data=df, ax=ax, s=5)

    if not points_only:
                    
        for x0, y0, patch in zip(df.pca_0.values, df.pca_1.values, patches):

            ab = AnnotationBbox(OffsetImage(patch, zoom=zoom), (x0, y0), frameon=False)

            ab.set_zorder(2)

            ax.add_artist(ab)
            
    plt.savefig(Path(directory_to_save_matplotlib_figures) / f'{figname}.png', dpi=150, bbox_inches='tight')
    
    return

########################## END OF VISUALIZATION OF EMBEDDED PATCHES ########################################