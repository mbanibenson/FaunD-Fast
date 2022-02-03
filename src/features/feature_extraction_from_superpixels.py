from functools import partial
from skimage.feature import hog
from skimage.transform import resize, rescale
import numpy as np
import tensorflow as tf
from skimage.io import imread
from pathlib import Path
import random

def extract_hand_engineered_hog_features_for_segmentation_patches(list_of_segment_patches):
    '''
    Given a list of segment patches as ndarrays, extract hand engineered features for each
    
    ARGUMENTS
    ---------
    list_of_segment_patches_as_ndarrays(list): List of superpixel patches as ndarrays
    
    
    RETURNS
    --------
    matrix_of_feature_vectors(ndarray): Feature vectors for each segment stacked as rows of a matrix
    
    '''
    model = partial(hog, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(4, 4),transform_sqrt=True, visualize=False, multichannel=True)
    
    print('Resizing images)')

    list_of_resized_ndarrays = [resize(im, (64,64)) for im in list_of_segment_patches]

    print('Extracting hog features ...')
    matrix_of_feature_vectors = np.vstack([model(image_array) for image_array in list_of_resized_ndarrays])
    
    return matrix_of_feature_vectors


def extract_convnet_features_for_segmentation_patches(feature_extractor_module_url, image_patches, resize_dimension):
    '''
    Extract convnet features using a pre-trained model from tf hub given an nd-array of image patches
    
    ARGUMENTS
    ----------
    feature_extractor_module_url(url): Path to feature extractor in tensorflow hub 
    
    image_patches(list): List of image patches from which to extract features 
    
    resize_dimension(tuple): Target image size expected by the feature extractor
    
    
    RETURN
    -------
    image_patches_feature_vectors(ndarray): Matrix of feature vectors extracted from the images
    
    '''
    #Resize to the dimension the tfhub module expects
    resized_image_patches = [np.expand_dims(resize(patch, resize_dimension, anti_aliasing=True), axis=0) for patch in image_patches]

    batch_of_images = np.concatenate(resized_image_patches, axis=0).astype(np.float32) #/ 255.

    model = tf.keras.Sequential([
    hub.KerasLayer(feature_extractor_module_url, trainable=False), # Can be True
    tf.keras.layers.GlobalAveragePooling2D(),
    ])
    
    image_patches_feature_vectors = model(batch_of_images[:500])
    
    
    return image_patches_feature_vectors


def extract_convnet_features_for_segmentation_patches_using_keras_applications(image_patches, resize_dimension=(224,224,3)):
    '''
    Extract convnet features
    
    '''
    
    feature_extractor = tf.keras.applications.ResNet50(
        include_top=False,
        pooling='avg',
    
    )

    resized_image_patches = [np.expand_dims(resize(patch, resize_dimension, anti_aliasing=True), axis=0) for patch in image_patches]

    batch_of_images = np.concatenate(resized_image_patches, axis=0).astype(np.float32)
    
    batch_of_images = tf.keras.applications.resnet.preprocess_input(batch_of_images)
    
    matrix_of_feature_vectors = feature_extractor(batch_of_images)
    
    return matrix_of_feature_vectors


def extract_hand_engineered_hog_support_set_feature_vectors(directory_containing_support_sets):
    '''
    Extract features from support set features contained within a directory
    
    ##TODO: Use Image Generator to load from different directories and label accordingly
    
    ARGUMENTS:
    ----------
    directory_containing_support_sets(path): Pathlike pointing to the directory containing the support set patches
    
    
    RETURN:
    ---------
    support_set_patches_feature_vectors(ndarray): Matrix in which the rows are the feature vectors extracted from the patches
    
    support_set_patches(ndarray): List of the image patches for the support sets
    
    '''
    support_set_patches = []
    
    support_set_labels = []
    
    for id_num, sub_directory  in enumerate(Path(directory_containing_support_sets).iterdir(), start=1):
        
        support_set_patches_in_this_subdirectory = [imread(fp) for fp in sub_directory.iterdir()]
        
        random.shuffle(support_set_patches_in_this_subdirectory)
        
        support_set_patches_in_this_subdirectory = support_set_patches_in_this_subdirectory
        
        labels_for_support_set_patches_in_this_subdirectory = [id_num] * len(support_set_patches_in_this_subdirectory)
        
        support_set_patches.extend(support_set_patches_in_this_subdirectory)
        
        support_set_labels.extend(labels_for_support_set_patches_in_this_subdirectory)
    
    # support_set_patches_feature_vectors = extract_hand_engineered_hog_features_for_segmentation_patches(support_set_patches)
    
    #support_set_labels = [1] * len(support_set_labels)
    
    support_set_patches_feature_vectors = extract_convnet_features_for_segmentation_patches_using_keras_applications(support_set_patches, resize_dimension=(224,224,3))
    
    

    return support_set_patches_feature_vectors, support_set_patches, support_set_labels