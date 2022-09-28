from pathlib import Path
from visualization.image_viewer_utils import create_post_processed_detections_summary_table
from visualization.image_viewer_utils import copy_files_to_image_viewer_for_annotation

if __name__ == '__main__':
    
    data_directory = Path.cwd().parents[0] / 'data'

    directory_containing_classification_outputs = data_directory / 'supervised_fauna_non_fauna_classification/classification_outputs'

    path_to_post_processed_summary_table = directory_containing_classification_outputs / 'master_detections_summary_table.csv'

    directory_containing_pure_fauna_patches = directory_containing_classification_outputs / 'classified_patches/fauna'
    
    directory_containing_unsupervised_outlier_detection_results = data_directory / 'unsupervised_outlier_detection'
    
    image_viewer_directory = Path.cwd().parents[0] / 'custom_annotation_tool'
    
    
    print('Saving post processed csv ...')
    create_post_processed_detections_summary_table(directory_containing_pure_fauna_patches, directory_containing_unsupervised_outlier_detection_results, path_to_post_processed_summary_table)
    
    
    print('Copying files to image viewer directory ...')
    copy_files_to_image_viewer_for_annotation(directory_containing_pure_fauna_patches, directory_containing_unsupervised_outlier_detection_results, path_to_post_processed_summary_table, image_viewer_directory)