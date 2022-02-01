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
from models.core_utils import segment_image_and_extract_segment_features

from features.metric_learning_utils import embedd_segment_feature_vectors_using_supervised_pca
from models.predict_model import run_inference_on_test_images
from visualization.visualize import visualize_embedded_segment_patches
from models.train_model import train_non_background_detection_model


rng = default_rng()

#if __name__ == '__main__':
    
    
directory_containing_underwater_images_with_background_only = Path('/home/mbani/mardata/datasets/Pacific_dataset/SO268-1_021-1_OFOS-02/')

directory_containing_support_sets = Path('/home/mbani/mardata/datasets/support set/')

directory_containing_test_images = Path('/home/mbani/mardata/datasets/fauna_images_from_all_dives/')


(training_embeddings, original_feature_vectors, training_embedding_labels, training_embedding_patches, 
optimization_results_object_for_finding_transformation_matrix, trained_pca) = train_non_background_detection_model(directory_containing_underwater_images_with_background_only, directory_containing_support_sets)


combined_embeddings, combined_labels, combined_patches = run_inference_on_test_images(directory_containing_test_images, training_embeddings, training_embedding_labels, training_embedding_patches, trained_pca)



visualize_embedded_segment_patches(training_embeddings, training_embedding_labels, figsize=(12,8))

visualize_embedded_segment_patches(training_embeddings, training_embeddings, training_embedding_patches, figsize=(12,8))

visualize_embedded_segment_patches(combined_embeddings, combined_labels, combined_patches, figsize=(12,8))