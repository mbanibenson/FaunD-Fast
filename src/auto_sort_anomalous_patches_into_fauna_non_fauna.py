from models.CNN_based_fauna_non_fauna_classifier import train_fauna_non_fauna_classifier_and_auto_sort_anomalous_patches
from pathlib import Path

if __name__ == '__main__':
    
    #Set variables
    target_size = 224
    number_of_classes = 2
    figsize = (12,8)
    dive_to_sort = 'random_dive'
    
    data_directory = Path.cwd().parents[0] / 'data'
    
    directory_containing_training_datasets = data_directory / 'supervised_fauna_non_fauna_classification/training_dataset/'
    
    directory_containing_classification_outputs = data_directory / 'supervised_fauna_non_fauna_classification/classification_outputs'
    
    shutil.rmtree(directory_containing_classification_outputs, ignore_errors=True)
    directory_containing_classification_outputs.mkdir()
    
    directory_to_save_sorted_images = directory_containing_classification_outputs / 'classified_patches'
    directory_to_save_sorted_images.mkdir()

    path_to_post_processed_summary_table = directory_containing_classification_outputs / 'master_detections_summary_table.csv'
    
    directory_to_save_matplotlib_figures = directory_containing_classification_outputs / 'matplotlib_figures'
    directory_to_save_matplotlib_figures.mkdir()
    
    directory_containing_pure_fauna_patches = directory_to_save_sorted_images / 'fauna'
    
    directory_containing_unsupervised_outlier_detection_results = data_directory / 'unsupervised_outlier_detection'
    
    
    train_fauna_non_fauna_classifier_and_auto_sort_anomalous_patches(directory_containing_training_datasets, target_size, number_of_classes, figsize, directory_to_save_matplotlib_figures, directory_containing_unsupervised_outlier_detection_results, directory_to_save_sorted_images)
    
    