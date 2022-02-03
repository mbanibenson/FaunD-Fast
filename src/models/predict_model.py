from pathlib import Path
import numpy as np
import random
from skimage.transform import resize, rescale
from .core_utils import merge_segmentation_patches_from_all_images
from .core_utils import segment_image_and_extract_segment_features
from itertools import compress
from scipy.spatial import ConvexHull
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from skimage.io import imsave
import shutil
from skimage.util import img_as_ubyte
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


def test_embeddings_and_return_outliers_using_bounding_envelope(test_embeddings, test_patches, hull):
    '''
    Process test points to determine outliers
    
    '''
    
    test_points = [Point(x,y) for x, y in test_embeddings]
    
    points_within_the_hull = [hull.contains(point) for point in test_points]
    
    points_outside_the_hull = [not elem for elem in points_within_the_hull]
    
    selector_for_outliers = points_outside_the_hull
    
    outlier_test_embeddings = np.compress(selector_for_outliers, test_embeddings, axis=0)
    
    outlier_test_patches = list(compress(test_patches, selector_for_outliers))
    
    outlier_test_labels = np.zeros(shape=(len(outlier_test_patches),))
    
    return outlier_test_embeddings, outlier_test_labels, outlier_test_patches


def run_inference_on_test_images(directory_containing_test_images, training_embeddings, training_embedding_labels, training_embedding_patches, trained_pca, novelty_detector, directory_to_save_patches_of_positive_detections, hull=None, feature_extractor_module_url=None, resize_dimension=None):
    '''
    Run inference on test images and return results for plotting and visualizations
    
    '''
    test_image_file_paths = random.sample(list(directory_containing_test_images.iterdir()), 30)
    
    # segmented_image_objects = [segment_image_and_extract_segment_features(fp, feature_extractor_module_url=feature_extractor_module_url, resize_dimension=resize_dimension) for fp in test_image_file_paths]
    
    with ProcessPoolExecutor(14) as executor:
        
        segmented_image_objects = list(executor.map(segment_image_and_extract_segment_features, test_image_file_paths))
        
    
    segmentation_feature_vectors, segment_patches, names_for_each_segment_patch, segment_patch_bboxes = merge_segmentation_patches_from_all_images(segmented_image_objects)
    
    segmentation_feature_vectors_labels = np.zeros(shape=(len(segmentation_feature_vectors),)) + 2

    segment_patches = [resize(patch, (96,96,3)) for patch in segment_patches]
    
    combined_patches = training_embedding_patches + segment_patches
    
    
    test_embeddings = trained_pca.transform(segmentation_feature_vectors)
    
    test_patches = segment_patches
    
    combined_embeddings = np.concatenate([training_embeddings, test_embeddings], axis=0)
    
    labels = np.concatenate([training_embedding_labels, segmentation_feature_vectors_labels])
    
    
    test_embeddings_outlier_or_inlier_prediction = novelty_detector.predict(test_embeddings)
    
    selector_for_outliers = test_embeddings_outlier_or_inlier_prediction == 1
    
    outlier_test_embeddings = np.compress(selector_for_outliers, test_embeddings, axis=0)
    
    outlier_test_patches = list(compress(segment_patches, selector_for_outliers))
    
    outlier_test_labels = [2] * len(outlier_test_patches)
    
    outlier_test_patch_names = list(compress(names_for_each_segment_patch, selector_for_outliers))
    
    outlier_test_patch_bboxes = list(compress(segment_patch_bboxes, selector_for_outliers))
    
    save_patches_to_directory(directory_to_save_patches_of_positive_detections, outlier_test_patches, outlier_test_patch_names)
    
    generate_csv_summarizing_detections(outlier_test_patch_names, outlier_test_embeddings, outlier_test_patch_bboxes, directory_to_save_patches_of_positive_detections)


    # outlier_test_embeddings, outlier_test_labels, outlier_test_patches = test_embeddings_and_return_outliers_using_bounding_envelope(test_embeddings, test_patches, hull)
    
    
    
    # outlier_test_embeddings, outlier_test_labels, outlier_test_patches = novelty_detector_using_bounding_envelope(background_embeddings, test_embeddings, test_patches)
    
    
    
    return outlier_test_embeddings, outlier_test_labels, outlier_test_patches


def save_patches_to_directory(directory_to_save_patches, patches, patch_names):
    '''
    Save detected pactches to disk
    
    '''
    directory_to_save_patches = Path(directory_to_save_patches)
    
    shutil.rmtree(directory_to_save_patches, ignore_errors=True)
    directory_to_save_patches.mkdir()
    
    print('Saving patches for positive detections ...')
    
    [imsave(directory_to_save_patches / f'{patch_name}.png', img_as_ubyte(patch)) for patch_name, patch in zip(patch_names, patches)]
    
    print('Finished saving patches for positive detections ...')
    
    return

def generate_csv_summarizing_detections(patch_names, patch_embeddings, patch_bboxes, directory_to_save_summary_csv):
    '''
    Summarize the detections into a csv file
    
    '''
    detections_summary = pd.DataFrame({'patch_name':patch_names, 'pca_1':patch_embeddings[:,0], 'pca_2':patch_embeddings[:,1], 'bbox':patch_bboxes})
    
    detections_summary['parent_image_name'] = detections_summary.patch_name.map(lambda x: x.split('#')[0])
    
    detections_summary.to_csv(directory_to_save_summary_csv/'detections_summary_table.csv')
    
    return