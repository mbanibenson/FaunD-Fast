from custom_object_detection.tf_object_detection_utilities import sample_ground_truth_images_for_annotation
import shutil
from pathlib import Path

if __name__ == '__main__':
    
    object_detection_working_directory = Path.cwd().parents[0] / 'fauna_detection_with_tensorflow_object_detection_api'
    
    evaluation_directory = object_detection_working_directory / 'performance_assessment'
    
    evaluation_directory.mkdir(exist_ok=True)
    
    directory_to_save_sampled_ground_truth_images = evaluation_directory / 'ground_truth_images'
    
    shutil.rmtree(directory_to_save_sampled_ground_truth_images, ignore_errors=True)
    
    directory_to_save_sampled_ground_truth_images.mkdir(exist_ok=True)
       
    path_to_annotations_table = object_detection_working_directory / 'data/object_detection_input_datasheet.csv'
    
    directory_with_all_parent_images = Path.cwd().parents[0] / 'custom_annotation_tool/parent_images'
    
    n_samples = 500
    
    print('Sampling images ...')
    
    sample_ground_truth_images_for_annotation(directory_with_all_parent_images, directory_to_save_sampled_ground_truth_images, path_to_annotations_table, n_samples)
    
    print('Finished.')