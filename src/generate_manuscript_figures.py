from visualization.manuscript_visualizations import generate_background_images_with_superpixel_overlays
from visualization.manuscript_visualizations import generate_grid_view_of_background_superpixels
from visualization.manuscript_visualizations import generate_feature_space_view_of_background_superpixels
from visualization.manuscript_visualizations import generate_feature_space_view_of_all_flagged_anomalous_superpixels
from parameters import deepsea_fauna_detection_params
import pickle
import shutil

if __name__ == '__main__':
    
    directory_with_pickled_items = deepsea_fauna_detection_params.EXAMPLE_DIRECTORY_WITH_PICKLED_ITEMS
    
    directory_to_save_manuscript_plots = deepsea_fauna_detection_params.MANUSCRIPT_FIGURES_DIRECTORY
    
    figsize = deepsea_fauna_detection_params.MANUSCRIPT_FIG_SIZE
    
    anomalous_superpixel_detection_output_directory = deepsea_fauna_detection_params.DIVE_OUTPUT_DIR
    
    print ('Creating manuscript figures directory ...')
    shutil.rmtree(directory_to_save_manuscript_plots, ignore_errors=True)
    directory_to_save_manuscript_plots.mkdir(exist_ok=True)
    
    ### Background Superpixel Boundaries
    path_to_pickled_segmented_images = directory_with_pickled_items / 'list_of_segmented_images.pickle'
    path_to_pickled_original_images = directory_with_pickled_items / 'list_of_original_images.pickle'  
    number_of_annotated_background_images = 10
    
    #generate_background_images_with_superpixel_overlays(path_to_pickled_segmented_images, path_to_pickled_original_images, directory_to_save_manuscript_plots, figsize, n_sample=number_of_annotated_background_images)
    
    ### Grid view of background superpixels
    path_to_pickled_background_patches = directory_with_pickled_items / 'list_of_training_patches.pickle'
    
    #generate_grid_view_of_background_superpixels(path_to_pickled_background_patches, directory_to_save_manuscript_plots, figsize)
    
    ### Feature space view of background superpixels
    path_to_pickled_background_feature_vectors = directory_with_pickled_items / 'background_feature_vectors.pickle'
    path_to_pickled_pca_object = directory_with_pickled_items / 'pca_object.pickle'
    
    generate_feature_space_view_of_background_superpixels(path_to_pickled_background_feature_vectors, path_to_pickled_background_patches, path_to_pickled_pca_object, directory_to_save_manuscript_plots, figsize)
    
    
    ### Feature space view of anomalous superpixels
    path_to_detections_summary_table = anomalous_superpixel_detection_output_directory / 'detections_summary_table.csv'
    directory_with_anomalous_superpixel_patches = anomalous_superpixel_detection_output_directory / 'patches'
    
    generate_feature_space_view_of_all_flagged_anomalous_superpixels(path_to_detections_summary_table, directory_with_anomalous_superpixel_patches, path_to_pickled_pca_object, directory_to_save_manuscript_plots, figsize)