from visualization.manuscript_visualizations import generate_background_images_with_superpixel_overlays
from visualization.manuscript_visualizations import generate_grid_view_of_background_superpixels
from visualization.manuscript_visualizations import generate_feature_space_view_of_background_superpixels
from visualization.manuscript_visualizations import generate_feature_space_view_of_all_flagged_anomalous_superpixels
from visualization.manuscript_visualizations import generate_feature_space_view_of_top_k_anomalous_superpixels
from visualization.manuscript_visualizations import generate_feature_space_view_of_anomalous_superpixels_after_binary_classification
from visualization.manuscript_visualizations import generate_grid_view_of_anomalous_superpixels_after_binary_classification
from visualization.manuscript_visualizations import generate_screenshot_of_superpixel_annotation_tool
from visualization.manuscript_visualizations import generate_distribution_of_annotated_morphotypes
from visualization.manuscript_visualizations import generate_distribution_of_detected_morphotypes
from visualization.manuscript_visualizations import generate_example_images_with_bounding_box_overlays

from parameters import deepsea_fauna_detection_params
import pickle
import shutil

if __name__ == '__main__':
    
    directory_with_pickled_items = deepsea_fauna_detection_params.EXAMPLE_DIRECTORY_WITH_PICKLED_ITEMS
    
    directory_to_save_manuscript_plots = deepsea_fauna_detection_params.MANUSCRIPT_FIGURES_DIRECTORY
    
    figsize = deepsea_fauna_detection_params.MANUSCRIPT_FIG_SIZE
    
    anomalous_superpixel_detection_output_directory = deepsea_fauna_detection_params.DIVE_OUTPUT_DIR
    
    anomalous_superpixel_detection_after_binary_classification_output_directory = deepsea_fauna_detection_params.DIVE_OUTPUT_DIR_AFTER_BINARY_CLASSIFICATION
    
    directory_with_annotation_tool = deepsea_fauna_detection_params.ANNOTATION_TOOL_DIR
    
    object_detection_directory = deepsea_fauna_detection_params.OBJECT_DETECTION_DIR
    
    print ('Creating manuscript figures directory ...')
    shutil.rmtree(directory_to_save_manuscript_plots, ignore_errors=True)
    directory_to_save_manuscript_plots.mkdir(exist_ok=True)
    
    ### Background Superpixel Boundaries
    path_to_pickled_segmented_images = directory_with_pickled_items / 'segmented_images.pickle'
    path_to_pickled_original_images = directory_with_pickled_items / 'original_images.pickle'  
    number_of_annotated_background_images = 10
    
    generate_background_images_with_superpixel_overlays(path_to_pickled_segmented_images, path_to_pickled_original_images, directory_to_save_manuscript_plots, figsize, n_sample=number_of_annotated_background_images)
    
    ### Grid view of background superpixels
    path_to_pickled_background_patches = directory_with_pickled_items / 'background_patches.pickle'
    grid_dimension = 20
    
    generate_grid_view_of_background_superpixels(path_to_pickled_background_patches, directory_to_save_manuscript_plots, grid_dimension, figsize)
    
    ### Feature space view of background superpixels
    path_to_pickled_background_feature_vectors = directory_with_pickled_items / 'background_feature_vectors.pickle'
    path_to_pickled_pca_object = directory_with_pickled_items / 'pca_object.pickle'
    
    generate_feature_space_view_of_background_superpixels(path_to_pickled_background_feature_vectors, path_to_pickled_background_patches, path_to_pickled_pca_object, directory_to_save_manuscript_plots, figsize)
    
    
    ### Feature space view of anomalous superpixels
    path_to_detections_summary_table = anomalous_superpixel_detection_output_directory / 'detections_summary_table.csv'
    directory_with_anomalous_superpixel_patches = anomalous_superpixel_detection_output_directory / 'patches'
    
    generate_feature_space_view_of_all_flagged_anomalous_superpixels(path_to_detections_summary_table, directory_with_anomalous_superpixel_patches, path_to_pickled_pca_object, directory_to_save_manuscript_plots, figsize)
    
    
    ### Feature space view of top-k anomalous superpixels
    path_to_detections_summary_table = anomalous_superpixel_detection_output_directory / 'detections_summary_table.csv'
    directory_with_anomalous_superpixel_patches = anomalous_superpixel_detection_output_directory / 'patches'
    k = 200
    
    generate_feature_space_view_of_top_k_anomalous_superpixels(path_to_detections_summary_table, directory_with_anomalous_superpixel_patches, path_to_pickled_pca_object, directory_to_save_manuscript_plots, figsize, k)
    
    
    ### Feature space view of post binary classifcation anomalous superpixels
    path_to_post_classification_detections_summary_table = anomalous_superpixel_detection_after_binary_classification_output_directory / 'master_detections_summary_table.csv'
    directory_with_post_classification_anomalous_superpixel_patches = anomalous_superpixel_detection_after_binary_classification_output_directory / 'classified_patches/fauna/'
    
    generate_feature_space_view_of_anomalous_superpixels_after_binary_classification(path_to_post_classification_detections_summary_table, directory_with_post_classification_anomalous_superpixel_patches, path_to_pickled_pca_object, directory_to_save_manuscript_plots, figsize)
    
    ### Grid view of anomalous superpixels
    
    generate_grid_view_of_anomalous_superpixels_after_binary_classification(directory_with_post_classification_anomalous_superpixel_patches, directory_to_save_manuscript_plots, grid_dimension, figsize)
    
    
    ### Screen shot of superpixel annotation software
    path_to_screenshot = directory_with_annotation_tool / 'screenshot/annotator.png'
    
    generate_screenshot_of_superpixel_annotation_tool(path_to_screenshot, directory_to_save_manuscript_plots, figsize)
    
    
    ### Generate distribution over annotated benthic megafauna
    path_to_annotated_datasheet = object_detection_directory / 'data/object_detection_input_datasheet.csv'
    
    generate_distribution_of_annotated_morphotypes(path_to_annotated_datasheet,directory_to_save_manuscript_plots, figsize)
    
    ### Generate distribution over detected benthic megafauna
    path_to_detection_summary_table = object_detection_directory / 'faster_rcnn_with_detection_checkpoint/predictions/detections_summary_table.csv'
    
    generate_distribution_of_detected_morphotypes(path_to_detection_summary_table, directory_to_save_manuscript_plots, figsize)
    
    
    ### Generate example detections with bbox overlays
    directory_with_example_detection_images = object_detection_directory / 'exemplar_figures/'
    
    generate_example_images_with_bounding_box_overlays(directory_with_example_detection_images, directory_to_save_manuscript_plots, figsize)