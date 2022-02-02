from sklearn.svm import OneClassSVM

import sys
sys.path.append('./')

from .core_utils import segment_image_and_extract_segment_features
from features.metric_learning_utils import embedd_segment_feature_vectors_using_supervised_pca
from features.feature_extraction_from_superpixels import extract_hand_engineered_hog_support_set_feature_vectors
from pathlib import Path
from scipy.spatial import ConvexHull
from shapely.geometry import Point, MultiPoint
from shapely.geometry.polygon import Polygon

def novelty_detector_using_bounding_envelope(background_embeddings):
    '''
    Use a convex hull around background patches as a decision function
    
    '''
    
    #hull = ConvexHull(background_embeddings)
    
    background_points = [Point(x,y) for x, y in background_embeddings]
    
    hull = MultiPoint(background_embeddings).convex_hull
    
    return hull



def train_non_background_detection_model(directory_containing_underwater_images_with_background_only, directory_containing_support_sets):
    '''
    Train a model to distinguish background from potential fauna superpixels
    
    '''
    underwater_images_file_paths = list(directory_containing_underwater_images_with_background_only.iterdir())[:10]

    underwater_images_of_ccz = [segment_image_and_extract_segment_features(file_path) for file_path in underwater_images_file_paths] ##TODO consider calling it process_segments


    support_set_feature_vectors, support_set_patches, support_set_labels = extract_hand_engineered_hog_support_set_feature_vectors(directory_containing_support_sets) ##TODO consider calling it process_support_sets


    embedded_feature_vectors, embedded_background_feature_vectors, labels, patches, optimization_results_object_for_finding_transformation_matrix, trained_pca = embedd_segment_feature_vectors_using_supervised_pca(underwater_images_of_ccz, support_set_feature_vectors, support_set_patches, support_set_labels)
    
    novelty_detector = None #fit_one_class_svm(embedded_background_feature_vectors)
    
    hull = novelty_detector_using_bounding_envelope(embedded_background_feature_vectors)
    
    return embedded_feature_vectors, embedded_background_feature_vectors, labels, patches, optimization_results_object_for_finding_transformation_matrix, trained_pca, novelty_detector, hull