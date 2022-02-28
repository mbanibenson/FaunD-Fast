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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from concurrent.futures import ProcessPoolExecutor
import random
from functools import partial
from sklearn.neighbors import KNeighborsClassifier

def novelty_detector_using_bounding_envelope(background_embeddings):
    '''
    Use a convex hull around background patches as a decision function
    
    '''
    
    #hull = ConvexHull(background_embeddings)
    
    background_embeddings = background_embeddings[:,[0,1]]
    
    background_points = [Point(x,y) for x, y in background_embeddings]
    
    hull = MultiPoint(background_embeddings).convex_hull
    
    return hull


def novelty_detector_using_binary_kernel_svm(training_embeddings, labels_for_training_embeddings):
    '''
    Train an svm classifier
    
    '''
    
    X = training_embeddings
    
    y = [1 if int(val) > 0 else int(val) for val in labels_for_training_embeddings]
    
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
    
    clf.fit(X, y)
    
    return clf


def novelty_detector_using_k_nearest_neighbors(training_embeddings, labels_for_training_embeddings):
    '''
    Train an knn classifier
    
    '''
    
    X = training_embeddings
    
    y = [val if int(val) == 0 else 1 for val in labels_for_training_embeddings]
    
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=14)
    
    knn.fit(X, y)
    
    return knn



def train_non_background_detection_model(directory_containing_underwater_images_with_background_only, directory_containing_support_sets):
    '''
    Train a model to distinguish background from potential fauna superpixels
    
    '''
    
    all_file_paths = list(directory_containing_underwater_images_with_background_only.iterdir())
    
    random.shuffle(all_file_paths)
    underwater_images_file_paths = all_file_paths #random.sample(all_file_paths, k=10)

    # underwater_images_of_ccz = [segment_image_and_extract_segment_features(file_path) for file_path in underwater_images_file_paths] ##TODO consider calling it process_segments
    
    with ProcessPoolExecutor(14) as executor:
        
        _segment_image_and_extract_segment_features = partial(segment_image_and_extract_segment_features, training_mode=True)
        
        underwater_images_of_ccz = list(executor.map(_segment_image_and_extract_segment_features, underwater_images_file_paths))


    print('Running optimization ...')
    embedded_feature_vectors, embedded_background_feature_vectors, labels, patches, nca, scaler = embedd_segment_feature_vectors_using_supervised_pca(underwater_images_of_ccz, directory_containing_support_sets)
    
    # print('Fitting svm novelty detector ...')
    # novelty_detector = novelty_detector_using_binary_kernel_svm(embedded_feature_vectors, labels) 
    
    #novelty_detector = fit_one_class_svm(embedded_background_feature_vectors)
    
    print('Fitting hull-based novelty detector ...')
    hull = novelty_detector_using_bounding_envelope(embedded_background_feature_vectors)
    
    print('Fitting knn novelty detector ...')
    novelty_detector = novelty_detector_using_k_nearest_neighbors(embedded_feature_vectors, labels)
    
    return embedded_feature_vectors, embedded_background_feature_vectors, labels, patches, nca, novelty_detector, hull, scaler