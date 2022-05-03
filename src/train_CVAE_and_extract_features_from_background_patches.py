from models.VAE_based_outlier_detection import create_tensorflow_dataset_from_numpy_ndarray
from models.VAE_based_outlier_detection import train_VAE
from models.VAE_based_outlier_detection import extract_features_using_trained_VAE
import tensorflow as tf
from parameters import deepsea_fauna_detection_params

import pickle

from parameters import deepsea_fauna_detection_params

if __name__ == '__main__':
    
    directory_containing_pickled_items = deepsea_fauna_detection_params.DIVE_PICKLED_ITEMS_DIR
    
    batch_size = deepsea_fauna_detection_params.BATCH_SIZE
    
    latent_dimension = deepsea_fauna_detection_params.LATENT_DIMENSION

    epochs = deepsea_fauna_detection_params.TRAINING_EPOCHS
    
    with open(directory_containing_pickled_items / f'list_of_training_patches.pickle', 'rb') as f:
        
        list_of_training_patches = pickle.load(f)
    
    print('Generating train/val tf dataset split of patches training images ...')
    train_generator, val_generator, number_of_train_batches, number_of_val_batches = create_tensorflow_dataset_from_numpy_ndarray(list_of_training_patches, batch_size, is_for_training=True)
    print('Finished creating training tf dataset ...')
    
    
    print('Training VAE ...')
    trained_VAE_encoder_model = train_VAE(latent_dimension, train_generator, val_generator, number_of_train_batches, number_of_val_batches, epochs = epochs)    
    print('Finished training VAE ...')
    
    
    print('Extracting features from background patches using trained VAE ...')
    background_feature_vectors = extract_features_using_trained_VAE(trained_VAE_encoder_model, train_generator, number_of_train_batches)
    print('Finished extracting features from background patches using trained VAE ...')
    
    
    print('Pickling features ...')
    model_file_path = str(directory_containing_pickled_items / 'trained_VAE_model')
    
    
    trained_VAE_encoder_model.save(model_file_path)
    
    with open(directory_containing_pickled_items / f'background_feature_vectors.pickle', 'wb') as f:
        
        pickle.dump(background_feature_vectors, f, pickle.HIGHEST_PROTOCOL)
        
    print('Done pickling features ...')