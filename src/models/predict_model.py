from pathlib import Path
import numpy as np
import random
from skimage.transform import resize, rescale
from .core_utils import merge_segmentation_patches_from_all_images
from .core_utils import segment_image_and_extract_segment_features
from itertools import compress


def run_inference_on_test_images(directory_containing_test_images, training_embeddings, training_embedding_labels, training_embedding_patches, trained_pca, novelty_detector, feature_extractor_module_url=None, resize_dimension=None):
    '''
    Run inference on test images and return results for plotting and visualizations
    
    '''
    test_image_file_paths = random.sample(list(directory_containing_test_images.iterdir()), 30)
    
    segmented_image_objects = [segment_image_and_extract_segment_features(fp, feature_extractor_module_url=feature_extractor_module_url, resize_dimension=resize_dimension) for fp in test_image_file_paths]
    
    segmentation_feature_vectors, segment_patches = merge_segmentation_patches_from_all_images(segmented_image_objects)
    
    segmentation_feature_vectors_labels = np.zeros(shape=(len(segmentation_feature_vectors),)) + 2

    segment_patches = [resize(patch, (96,96,3)) for patch in segment_patches]
    
    combined_patches = training_embedding_patches + segment_patches
    
    
    test_embeddings = trained_pca.transform(segmentation_feature_vectors)
    
    combined_embeddings = np.concatenate([training_embeddings, test_embeddings], axis=0)
    
    labels = np.concatenate([training_embedding_labels, segmentation_feature_vectors_labels])
    
    
    test_embeddings_outlier_or_inlier_prediction = novelty_detector.predict(test_embeddings)
    
    selector_for_outliers = test_embeddings_outlier_or_inlier_prediction == -1
    
    outlier_test_embeddings = np.compress(selector_for_outliers, test_embeddings)
    
    outlier_test_patches = compress(segment_patches, selector_for_outliers)
    
    outlier_test_labels = np.zeros(shape=(len(outlier_test_patches),))
    
    
    return outlier_test_embeddings, outlier_test_labels, outlier_test_patches