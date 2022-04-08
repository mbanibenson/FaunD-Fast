from pathlib import Path
import numpy as np
import pandas as pd
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
from math import ceil
from functools import partial
from scipy.ndimage import zoom
from concurrent.futures import ProcessPoolExecutor
from skimage.exposure import rescale_intensity
from features.feature_extraction_from_superpixels import extract_SIFT_features_for_segmentation_patches_using_kornia


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
    
    return selector_for_outliers


# def run_inference_on_test_images(directory_containing_test_images, training_embeddings, training_embedding_labels, training_embedding_patches, trained_pca, novelty_detector, directory_to_save_patches_of_positive_detections, hull=None, feature_extractor_module_url=None, resize_dimension=None):
#     '''
#     Run inference on test images and return results for plotting and visualizations
    
#     '''
#     test_image_file_paths = list(directory_containing_test_images.iterdir())
#     #test_image_file_paths = random.sample(test_image_file_paths, 30)
    
#     # segmented_image_objects = [segment_image_and_extract_segment_features(fp, feature_extractor_module_url=feature_extractor_module_url, resize_dimension=resize_dimension) for fp in test_image_file_paths]
    
#     with ProcessPoolExecutor(14) as executor:
        
#         segmented_image_objects = list(executor.map(segment_image_and_extract_segment_features, test_image_file_paths))
        
    
#     segmentation_feature_vectors, segment_patches, names_for_each_segment_patch, segment_patch_bboxes = merge_segmentation_patches_from_all_images(segmented_image_objects)
    
#     segmentation_feature_vectors_labels = np.zeros(shape=(len(segmentation_feature_vectors),)) + 2

#     #segment_patches = [resize(patch, (96,96,3)) for patch in segment_patches]
    
#     combined_patches = training_embedding_patches + segment_patches
    
    
#     test_embeddings = trained_pca.transform(segmentation_feature_vectors)
    
#     test_patches = segment_patches
    
#     combined_embeddings = np.concatenate([training_embeddings, test_embeddings], axis=0)
    
#     labels = np.concatenate([training_embedding_labels, segmentation_feature_vectors_labels])
    
    
#     test_embeddings_outlier_or_inlier_prediction = novelty_detector.predict(test_embeddings)
    
#     selector_for_outliers = test_embeddings_outlier_or_inlier_prediction == 1
    
#     outlier_test_embeddings = np.compress(selector_for_outliers, test_embeddings, axis=0)
    
#     outlier_test_patches = list(compress(segment_patches, selector_for_outliers))
    
#     outlier_test_labels = [2] * len(outlier_test_patches)
    
#     outlier_test_patch_names = list(compress(names_for_each_segment_patch, selector_for_outliers))
    
#     outlier_test_patch_bboxes = list(compress(segment_patch_bboxes, selector_for_outliers))
    
#     save_patches_to_directory(directory_to_save_patches_of_positive_detections, outlier_test_patches, outlier_test_patch_names)
    
#     generate_csv_summarizing_detections(outlier_test_patch_names, outlier_test_embeddings, outlier_test_patch_bboxes, directory_to_save_patches_of_positive_detections)


#     # outlier_test_embeddings, outlier_test_labels, outlier_test_patches = test_embeddings_and_return_outliers_using_bounding_envelope(test_embeddings, test_patches, hull)
    
    
    
#     # outlier_test_embeddings, outlier_test_labels, outlier_test_patches = novelty_detector_using_bounding_envelope(background_embeddings, test_embeddings, test_patches)
    
    
    
#     return outlier_test_embeddings, outlier_test_labels, outlier_test_patches


# def run_inference_on_test_images(directory_containing_test_images, training_embeddings, training_embedding_labels, training_embedding_patches, trained_nca, pca, novelty_detector, directory_to_save_patches_of_positive_detections, hull=None, feature_extractor_module_url=None, resize_dimension=None):
#     '''
#     Run inference on test images and return results for plotting and visualizations
    
#     '''
#     test_image_file_paths = list(directory_containing_test_images.iterdir())
#     #test_image_file_paths = random.sample(test_image_file_paths, 30)
    
#     shutil.rmtree(directory_to_save_patches_of_positive_detections, ignore_errors=True)
#     directory_to_save_patches_of_positive_detections.mkdir()
    
#     number_of_partitions = ceil(len(test_image_file_paths) / 20)
    
#     test_image_file_paths_partitions = np.array_split(np.asarray(test_image_file_paths), number_of_partitions)
    
#     outlier_test_patch_names_for_all_partitions = [] 
    
#     outlier_test_embeddings_for_all_partitions = []
    
#     outlier_test_patch_bboxes_for_all_partitions = []
    
#     outlier_test_patches_for_all_partitions = []
    
#     for partition_id, partition_of_file_paths in enumerate(test_image_file_paths_partitions):
        
#         print(f'[INFO] Processing partition {partition_id} / {number_of_partitions} ...')
    
#         with ProcessPoolExecutor(14) as executor:
            
#             _segment_image_and_extract_segment_features = partial(segment_image_and_extract_segment_features, training_mode=False)

#             segmented_image_objects = list(executor.map(_segment_image_and_extract_segment_features, partition_of_file_paths))
        
    
#         segmentation_feature_vectors, segment_patches, names_for_each_segment_patch, segment_patch_bboxes = merge_segmentation_patches_from_all_images(segmented_image_objects)
    
#         #segmentation_feature_vectors_labels = np.zeros(shape=(len(segmentation_feature_vectors),)) + 2

#         #segment_patches = [resize(patch, (96,96,3)) for patch in segment_patches]
    
    
#         test_embeddings = trained_nca.transform(pca.transform(segmentation_feature_vectors))
    
#         test_patches = segment_patches

#         test_embeddings_outlier_or_inlier_prediction = novelty_detector.predict(test_embeddings)
        
# #         # test_embeddings_classification_probabilities = np.amax(novelty_detector.predict_proba(test_embeddings), axis=1)
        
        
#         selector_for_classification = test_embeddings_outlier_or_inlier_prediction == 1 
        
#         selector_for_outliers = selector_for_classification
        
        
# #         selector_for_probability = test_embeddings_classification_probabilities > 0.8
        
# #         selector_for_outliers = np.logical_and(selector_for_classification, selector_for_probability)
        
#         # selector_for_outliers = test_embeddings_and_return_outliers_using_bounding_envelope(test_embeddings, test_patches, hull)
    
        
    
#         outlier_test_embeddings = np.compress(selector_for_outliers, test_embeddings, axis=0)
    
#         outlier_test_patches = list(compress(segment_patches, selector_for_outliers))
    
#         outlier_test_patch_names = list(compress(names_for_each_segment_patch, selector_for_outliers))
    
#         outlier_test_patch_bboxes = list(compress(segment_patch_bboxes, selector_for_outliers))
    
#         save_patches_to_directory(directory_to_save_patches_of_positive_detections, outlier_test_patches, outlier_test_patch_names)
        
#         outlier_test_patch_names_for_all_partitions.extend(outlier_test_patch_names)
        
#         outlier_test_patch_bboxes_for_all_partitions.extend(outlier_test_patch_bboxes)
        
#         outlier_test_embeddings_for_all_partitions.append(outlier_test_embeddings)
        
#         outlier_test_patches_for_all_partitions.extend(outlier_test_patches)
        
#     outlier_test_embeddings_for_all_partitions = np.concatenate(outlier_test_embeddings_for_all_partitions, axis=0)
    
#     generate_csv_summarizing_detections(outlier_test_patch_names_for_all_partitions, outlier_test_embeddings_for_all_partitions, outlier_test_patch_bboxes_for_all_partitions, directory_to_save_patches_of_positive_detections)
    
#     outlier_test_labels_for_all_partitions = [2] * len(outlier_test_patches_for_all_partitions)


#     # outlier_test_embeddings, outlier_test_labels, outlier_test_patches = test_embeddings_and_return_outliers_using_bounding_envelope(test_embeddings, test_patches, hull)
    
    
    
#     # outlier_test_embeddings, outlier_test_labels, outlier_test_patches = novelty_detector_using_bounding_envelope(background_embeddings, test_embeddings, test_patches)
    
    
    
#     return outlier_test_embeddings_for_all_partitions, outlier_test_labels_for_all_partitions, outlier_test_patches_for_all_partitions



def run_inference_on_test_images(directory_containing_test_images, training_embeddings, training_embedding_labels, training_embedding_patches, trained_nca, novelty_detector, directory_to_save_patches_of_positive_detections, scaler, pca_for_visualization,label_encoder, hull=None):
    '''
    Run inference on test images and return results for plotting and visualizations
    
    '''
    test_image_file_paths = list(directory_containing_test_images.iterdir())
    #test_image_file_paths = random.sample(test_image_file_paths, 30)
    
    shutil.rmtree(directory_to_save_patches_of_positive_detections, ignore_errors=True)
    directory_to_save_patches_of_positive_detections.mkdir()
    
    number_of_partitions = ceil(len(test_image_file_paths) / 20)
    
    test_image_file_paths_partitions = np.array_split(np.asarray(test_image_file_paths), number_of_partitions)
    
    outlier_test_patch_names_for_all_partitions = [] 
    
    outlier_test_embeddings_for_all_partitions = []
    
    outlier_test_patch_bboxes_for_all_partitions = []
    
    outlier_test_patches_for_all_partitions = []
    
    outlier_test_patch_class_labels_for_all_partitions = []
    
    outlier_test_patch_prediction_probabilities_for_all_partitions = []
    
    for partition_id, partition_of_file_paths in enumerate(test_image_file_paths_partitions):
        
        print(f'[INFO] Processing partition {partition_id} / {number_of_partitions} ...')
    
        with ProcessPoolExecutor(14) as executor:
            
            _segment_image_and_extract_segment_features = partial(segment_image_and_extract_segment_features, training_mode=False)

            segmented_image_objects = list(executor.map(_segment_image_and_extract_segment_features, partition_of_file_paths))
        
    
        segment_patches, names_for_each_segment_patch, segment_patch_bboxes, _ = merge_segmentation_patches_from_all_images(segmented_image_objects)
        
        segment_patches_as_matrix = np.vstack([patch.ravel() for patch in segment_patches])

        segment_patches_standardized = scaler.transform(segment_patches_as_matrix)


        #Reshape them back to original dimension
        segment_patches_standardized = [flattened_patch.reshape((32,32,3)) for flattened_patch in segment_patches_standardized]
        
        segmentation_feature_vectors = extract_SIFT_features_for_segmentation_patches_using_kornia(segment_patches_standardized)

        test_embeddings = trained_nca.transform(segmentation_feature_vectors)
    
        test_patches = segment_patches

        test_embeddings_predictions_as_integers = novelty_detector.predict(test_embeddings)
        
        test_embeddings_predictions_with_class_names = label_encoder.inverse_transform(test_embeddings_predictions_as_integers)
        
        selector_for_classification = ~pd.Series(test_embeddings_predictions_with_class_names).str.startswith('back')
        
        selector_for_classification = selector_for_classification.tolist()
        
        test_embeddings_predictions_probabilities = np.amax(novelty_detector.predict_proba(test_embeddings), axis=1)
        
        
        # selector_for_classification = np.where(test_embeddings_outlier_or_inlier_prediction != 0, True, False)
        
        selector_for_outliers = selector_for_classification
        
        
#         selector_for_probability = test_embeddings_classification_probabilities > 0.8
        
#         selector_for_outliers = np.logical_and(selector_for_classification, selector_for_probability)
        
        # selector_for_outliers = test_embeddings_and_return_outliers_using_bounding_envelope(test_embeddings, test_patches, hull)
    
        
    
        outlier_test_embeddings = np.compress(selector_for_outliers, test_embeddings, axis=0)
    
        outlier_test_patches = list(compress(segment_patches, selector_for_outliers))
    
        outlier_test_patch_names = list(compress(names_for_each_segment_patch, selector_for_outliers))
    
        outlier_test_patch_bboxes = list(compress(segment_patch_bboxes, selector_for_outliers))
        
        outlier_test_patch_class_labels = list(compress(test_embeddings_predictions_with_class_names, selector_for_outliers))
        
        test_embeddings_predictions_probabilities = list(compress(test_embeddings_predictions_probabilities, selector_for_outliers))
    
        save_patches_to_directory(directory_to_save_patches_of_positive_detections, outlier_test_patches, outlier_test_patch_names, outlier_test_patch_class_labels)
        
        outlier_test_patch_names_for_all_partitions.extend(outlier_test_patch_names)
        
        outlier_test_patch_bboxes_for_all_partitions.extend(outlier_test_patch_bboxes)
        
        outlier_test_embeddings_for_all_partitions.append(outlier_test_embeddings)
        
        outlier_test_patches_for_all_partitions.extend(outlier_test_patches)
        
        outlier_test_patch_class_labels_for_all_partitions.extend(outlier_test_patch_class_labels)
        
        outlier_test_patch_prediction_probabilities_for_all_partitions.extend(test_embeddings_predictions_probabilities)
        
        
    outlier_test_embeddings_for_all_partitions = np.concatenate(outlier_test_embeddings_for_all_partitions, axis=0)
    
    outlier_test_embeddings_for_all_partitions_in_2d = pca_for_visualization.transform(outlier_test_embeddings_for_all_partitions)
    
    generate_csv_summarizing_detections(outlier_test_patch_names_for_all_partitions, outlier_test_embeddings_for_all_partitions, outlier_test_patch_bboxes_for_all_partitions, outlier_test_embeddings_for_all_partitions_in_2d, outlier_test_patch_class_labels_for_all_partitions,outlier_test_patch_prediction_probabilities_for_all_partitions, directory_to_save_patches_of_positive_detections)
    
    outlier_test_labels_for_all_partitions = [2] * len(outlier_test_patches_for_all_partitions)


    # outlier_test_embeddings, outlier_test_labels, outlier_test_patches = test_embeddings_and_return_outliers_using_bounding_envelope(test_embeddings, test_patches, hull)
    
    
    
    # outlier_test_embeddings, outlier_test_labels, outlier_test_patches = novelty_detector_using_bounding_envelope(background_embeddings, test_embeddings, test_patches)
    
    
    
    return outlier_test_embeddings_for_all_partitions_in_2d, outlier_test_labels_for_all_partitions, outlier_test_patches_for_all_partitions

def patch_save_utility(directory_to_save, patch_file_name, patch_array):
    '''
    Utility function for saving patch names
    
    '''
    imsave(directory_to_save / f'{patch_file_name}.png', zoom(img_as_ubyte(patch_array),(3,3,1)))
    

def save_patches_to_directory(directory_to_save_patches, patches, patch_names):
    '''
    Save detected pactches to disk
    
    '''
    directory_to_save_patches = Path(directory_to_save_patches) / 'patches'
    
    directory_to_save_patches.mkdir(exist_ok=True)
       
    print('Saving patches for positive detections ...')
    
    try:
    
        #[imsave(directory_to_save_patches / f'{patch_class}/{patch_name}.png', zoom(img_as_ubyte(patch),(3,3,1))) for patch_class, patch_name, patch in zip(patch_class_labels, patch_names, patches)]
        with ProcessPoolExecutor(14) as executor:
            
            [executor.submit(patch_save_utility, directory_to_save_patches, patch_name, patch) for patch_name, patch in zip(patch_names, patches)]
    
    except:
        
        pass
    
    print('Finished saving patches for positive detections ...')
    
    return

def generate_csv_summarizing_detections(patch_names, patch_embeddings, patch_bboxes, outlier_test_embeddings_for_all_partitions_in_2d, patch_prediction_probabilities, directory_to_save_summary_csv):
    '''
    Summarize the detections into a csv file
    
    '''
    # dataframe_contents = {'patch_name':patch_names,'bbox':patch_bboxes, 'class_label':patch_class_labels, 'prediction_probability':patch_prediction_probabilities}
    
    dataframe_contents = {'patch_name':patch_names,'bbox':patch_bboxes, 'anomaly_score':patch_prediction_probabilities}
        
    detections_summary = pd.DataFrame(dataframe_contents)
    
    
    detections_summary['parent_image_name'] = detections_summary.patch_name.map(lambda x: x.split('#')[0])
    
    #Reformat bbox to agree with image viewer
    bbox_reformated = detections_summary['bbox'].map(lambda x: str(x).strip('()').replace(', ', '-'))
    
    detections_summary['bbox_original_format'] = detections_summary['bbox']

    detections_summary['bbox'] = bbox_reformated
    
    detections_summary = detections_summary.loc[:,['patch_name', 'parent_image_name', 'anomaly_score', 'bbox', 'bbox_original_format']]
        
    #Add the feature vectors
    for i in range(patch_embeddings.shape[1]):
        
        detections_summary[f'feature_{i}'] = patch_embeddings[:,i]
        
    for i in range(outlier_test_embeddings_for_all_partitions_in_2d.shape[1]):
        
        detections_summary[f'pca_{i}'] = outlier_test_embeddings_for_all_partitions_in_2d[:,i]
                    
    #Sort by anomaly score
    detections_summary = detections_summary.sort_values(by='anomaly_score', ascending=True)
        
    detections_summary.to_csv(directory_to_save_summary_csv/'detections_summary_table.csv', index=False)
    
    return