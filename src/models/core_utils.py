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

import sys
sys.path.append('./')

from data.read_datasets import read_individual_rgb_image
from features.superpixel_generation import generate_superpixels_using_slic
from features.superpixel_generation import extract_image_patches_corresponding_to_the_superpixels
from features.feature_extraction_from_superpixels import extract_hand_engineered_hog_features_for_segmentation_patches

from features.feature_extraction_from_superpixels import extract_convnet_features_for_segmentation_patches_using_keras_applications

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
        
        self.segment_patch_bounding_boxes = None
        
        self.identifier_name_for_each_patch = None
        
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
    
    
    def extract_segmentation_patches_to_batch_of_ndarrays(self, training_mode):
        '''
        Extract segment patches to numpy ndarray
        
        #features/segment_to_generate_superpixels.py
        segment_patches = extract_image_patches_corresponding_to_the_superpixels(segmented_image, image_as_rgb)
        
        '''
        segmented_image=self.segmented_image 
        
        image_as_rgb=self.rgb_image

        self.segment_patches, self.segment_patch_bounding_boxes = extract_image_patches_corresponding_to_the_superpixels(segmented_image, image_as_rgb, training_mode)
        
        self.identifier_name_for_each_patch = [f'{self.image_path.stem}#{patch_id}' for patch_id in range(len(self.segment_patches))]

        return
    
    
    def extract_features_from_segmentation_patches(self, feature_extractor_module_url=None, resize_dimension=None):
        '''
        Extract feature vectors from segmentation patches
        
        #features/segment_to_generate_superpixels.py
        extract_hand_engineered_hog_features_for_segmentation_patches(list_of_segment_patches)
        '''
        list_of_segment_patches = self.segment_patches

        self.segment_patches_feature_vectors = extract_hand_engineered_hog_features_for_segmentation_patches(list_of_segment_patches)
        
        # self.segment_patches_feature_vectors = extract_convnet_features_for_segmentation_patches_using_keras_applications(list_of_segment_patches, resize_dimension=(224,224,3))

        return
    
    #################################### END OF CLASS METHODS #########################################
    
    
    

def segment_image_and_extract_segment_features(file_path, training_mode, feature_extractor_module_url=None, resize_dimension=None):
    '''
    Create an instance of underwater, segment it and extract features from its superpixels
    
    '''
    
    #print(f'Processing image {file_path.name}', end='\r')
    
    print('Creating segmentation object ...')
    underwater_image_of_ccz = underwater_image(file_path) #ccz is the working area in the pacific
    
    print('Reading image to array ...')
    underwater_image_of_ccz.read_image()
    
    print('Segmenting the image ...')
    underwater_image_of_ccz.segment_image()
    
    print('Converting segment patches to ndarrays ...')
    underwater_image_of_ccz.extract_segmentation_patches_to_batch_of_ndarrays(training_mode)
    
    print('Extract Features from the segments ...')
    underwater_image_of_ccz.extract_features_from_segmentation_patches(feature_extractor_module_url, resize_dimension)
    
    
    return underwater_image_of_ccz



def merge_segmentation_patches_from_all_images(segmented_image_objects):
    '''
    Gather all feature vectors from all instances of segmented images
    
    '''
    feature_vectors = np.concatenate([segmented_image_object.segment_patches_feature_vectors for segmented_image_object in segmented_image_objects], axis=0)
    
    segment_patches = list(chain.from_iterable(segmented_image_object.segment_patches for segmented_image_object in segmented_image_objects))
    
    segment_patch_names = list(chain.from_iterable(segmented_image_object.identifier_name_for_each_patch for segmented_image_object in segmented_image_objects))
    
    segment_patch_bboxes = list(chain.from_iterable(segmented_image_object.segment_patch_bounding_boxes for segmented_image_object in segmented_image_objects))
    
    assert len(segment_patch_names) == len(segment_patches), 'Number of patches does not match their labels'
    
    return feature_vectors, segment_patches, segment_patch_names, segment_patch_bboxes