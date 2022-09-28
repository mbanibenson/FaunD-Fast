from models.VAE_based_outlier_detection import create_tensorflow_dataset_from_numpy_ndarray
from models.VAE_based_outlier_detection import train_VAE
from models.VAE_based_outlier_detection import extract_features_using_trained_VAE
from sklearn.decomposition import KernelPCA
import tensorflow as tf
from parameters import deepsea_fauna_detection_params
from sklearn.decomposition import PCA
import pickle
from keras.models import load_model

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
    train_VAE(latent_dimension, train_generator, val_generator, number_of_train_batches, number_of_val_batches, directory_containing_pickled_items, epochs = epochs)    
    print('Finished training VAE ...')
    
    
    print('Generating test tf dataset split of patches test images ...')
    test_generator, number_of_test_batches = create_tensorflow_dataset_from_numpy_ndarray(list_of_training_patches, batch_size, is_for_training=False)
    print('Finished creating test tf dataset ...')    
    
    print('Extracting features from background patches using trained VAE ...')
    VAE_model_file_path = directory_containing_pickled_items / 'trained_VAE_model'
    
    trained_VAE_encoder_model = load_model(VAE_model_file_path, compile=False)

    background_feature_vectors = extract_features_using_trained_VAE(trained_VAE_encoder_model, test_generator, number_of_test_batches)
    
    print('Finished extracting features from background patches using trained VAE ...')
    
    print('Fitting pca object ...')
    pca_object = PCA(n_components=2)
    pca_object.fit(background_feature_vectors)
    
    
#     print('Pickling VAE model ...')
#     model_file_path = str(directory_containing_pickled_items / 'trained_VAE_model')
    
#     trained_VAE_encoder_model.save(model_file_path)
    
    print('Pickling feature vectors ...')
    with open(directory_containing_pickled_items / f'background_feature_vectors.pickle', 'wb') as f:
        
        pickle.dump(background_feature_vectors, f, pickle.HIGHEST_PROTOCOL)
        
    print('Pickling pca ...')
    with open(directory_containing_pickled_items / f'pca_object.pickle', 'wb') as f:
        
        pickle.dump(pca_object, f, pickle.HIGHEST_PROTOCOL)
        
    print('Done pickling things ...')