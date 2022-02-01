from pathlib import Path
from skimage.io import imread
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
import numpy as np
from skimage.transform import resize, rescale
from sklearn.neighbors import NeighborhoodComponentsAnalysis
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy.random import default_rng
#import tensorflow_hub as hub
import tensorflow as tf
from itertools import chain
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import KernelPCA
from skimage.feature import hog, BRIEF
from functools import partial
from skimage.color import rgb2gray
import  random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import norm
from scipy.optimize import nnls
from scipy.optimize import minimize

#Modules
from data.read_datasets import read_individual_rgb_image
from features.superpixel_generation import generate_superpixels_using_slic
from features.superpixel_generation import extract_image_patches_corresponding_to_the_superpixels
from features.feature_extraction_from_superpixels import extract_hand_engineered_hog_features_for_segmentation_patches
from features.feature_extraction_from_superpixels import extract_hand_engineered_hog_support_set_feature_vectors
from features.metric_learning_utils import embedd_segment_feature_vectors_using_supervised_pca
from models.predict_model import run_inference_on_test_images
from visualization.visualize import visualize_embedded_segment_patches


rng = default_rng()



class underwater_image:
    '''
    Read an image and segment it
    
    '''
    def __init__(self, path_to_image):
        
        self.image_path = Path(path_to_image)
        
        self.rgb_image = None
        
        self.segmented_image = None
        
        self.segment_patches = None
        
        self.segment_patches_feature_vectors = None
        
        self.segment_patch_centroids = None
        
        self.georeferenced_coordinates_for_the_image = None
        
        
    def read_image(self):
        '''
        Load image from disk
        
        #data/read_datasets.py
        rescaled_image = read_individual_rgb_image(file_path, scaling_factors=None)
        
        '''
        file_path = self.image_path
        
        scaling_factors = (0.25,0.25,1)

        self.rgb_image = read_individual_rgb_image(file_path, scaling_factors=scaling_factors)
        
        return
    
    
    def segment_image(self):
        '''
        Perform segmentation on the image
        
        #features/segment_to_generate_superpixels.py
        segmented_image = generate_superpixels_using_slic(image_as_rgb, number_of_segments, compactness)
        
        '''
        image_as_rgb=self.rgb_image
        
        number_of_segments=250
        
        compactness=50

        self.segmented_image = generate_superpixels_using_slic(image_as_rgb, number_of_segments, compactness)
        
        return
    
    
    def extract_segmentation_patches_to_batch_of_ndarrays(self):
        '''
        Extract segment patches to numpy ndarray
        
        #features/segment_to_generate_superpixels.py
        segment_patches = extract_image_patches_corresponding_to_the_superpixels(segmented_image, image_as_rgb)
        
        '''
        segmented_image=self.segmented_image, 
        
        image_as_rgb=self.rgb_image

        self.segment_patches = extract_image_patches_corresponding_to_the_superpixels(segmented_image, image_as_rgb)

        return
    
    
    def extract_features_from_segmentation_patches(self, feature_extractor_module_url=None, resize_dimension=None):
        '''
        Extract feature vectors from segmentation patches
        
        #features/segment_to_generate_superpixels.py
        extract_hand_engineered_hog_features_for_segmentation_patches(list_of_segment_patches)
        '''
        list_of_segment_patches = self.segment_patches

        self.segment_patches_feature_vectors = extract_hand_engineered_hog_features_for_segmentation_patches(list_of_segment_patches)

        return
    
    #################################### END OF CLASS METHODS #########################################
    
    
    

def segment_image_and_extract_segment_features(file_path, feature_extractor_module_url=None, resize_dimension=None):
    '''
    Create an instance of underwater, segment it and extract features from its superpixels
    
    '''
    
    #print(f'Processing image {file_path.name}', end='\r')
    
    print('Creating segmentation object ...', end='\r', flush=True)
    underwater_image_of_ccz = underwater_image(file_path) #ccz is the working area in the pacific
    
    print('Reading image to array ...', end='\r', flush=True)
    underwater_image_of_ccz.read_image()
    
    print('Segmenting the image ...', end='\r', flush=True)
    underwater_image_of_ccz.segment_image()
    
    print('Converting segment patches to ndarrays ...', end='\r', flush=True)
    underwater_image_of_ccz.extract_segmentation_patches_to_batch_of_ndarrays()
    
    print('Extract Features from the segments ...', end='\r', flush=True)
    underwater_image_of_ccz.extract_features_from_segmentation_patches(feature_extractor_module_url, resize_dimension)
    
    
    return underwater_image_of_ccz


if __name__ == '__main__':
    
    
    directory_containing_underwater_images_with_background_only = Path('/home/mbani/mardata/datasets/Pacific_dataset/SO268-1_021-1_OFOS-02/')
    
    directory_containing_support_sets = Path('/home/mbani/mardata/datasets/support set/')
    
    directory_containing_test_images = Path('/home/mbani/mardata/datasets/fauna_images_from_all_dives/')
    
    
    
    underwater_images_file_paths = list(directory_containing_underwater_images.iterdir())[:10]
    
    underwater_images_of_ccz = [segment_image_and_extract_segment_features(file_path) for file_path in underwater_images_file_paths]
    
    
    support_set_feature_vectors, support_set_patches = extract_hand_engineered_hog_support_set_feature_vectors(directory_containing_support_sets)
    
    

    embedded_feature_vectors, original_feature_vectors, labels, patches, optimization_results_object_for_finding_transformation_matrix, trained_pca = embedd_segment_feature_vectors_using_supervised_pca(underwater_images_of_ccz, support_set_feature_vectors, support_set_patches)
    
    
    test_image_file_paths = random.sample(list(directory_containing_test_images.iterdir()), 30)
    
    training_embeddings =  embedded_feature_vectors

    training_embedding_labels = labels

    training_embedding_patches = patches
    
    combined_embeddings, combined_labels, combined_patches = run_inference_on_test_images(test_image_file_paths, training_embeddings, training_embedding_labels, training_embedding_patches, trained_pca, feature_extractor_module_url, resize_dimension)




    visualize_embedded_segment_patches(embedded_feature_vectors, labels, figsize=(12,8))
    
    visualize_embedded_segment_patches(embedded_feature_vectors, labels, patches, figsize=(12,8))
    
    visualize_embedded_segment_patches(combined_embeddings, combined_labels, combined_patches, figsize=(12,8))