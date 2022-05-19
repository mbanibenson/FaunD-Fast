from custom_object_detection.tf_object_detection_utilities import load_saved_model_for_inference
from custom_object_detection.tf_object_detection_utilities import detect_objects_in_image
from custom_object_detection.tf_object_detection_utilities import load_label_map_info
from custom_object_detection.tf_object_detection_utilities import visualize_detections
from custom_object_detection.tf_object_detection_utilities import read_image_into_a_tensor
from custom_object_detection.tf_object_detection_utilities import generate_matrix_of_detections
from custom_object_detection.tf_object_detection_utilities import save_georeferenced_detection_results_as_csv
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
import time

if __name__ == '__main__':
    
    object_detection_working_directory = Path.cwd().parents[0] / 'fauna_detection_with_tensorflow_object_detection_api'
    
    path_to_saved_model = object_detection_working_directory / 'my_model_dir/exported_model_dir/saved_model'
    
    label_map_path = object_detection_working_directory / 'data/SO268_label_map.pbtxt'
    
    directory_with_subdirectories_to_detect = Path.cwd().parents[0] /'data/unsupervised_outlier_detection'
    
    directory_to_save_predictions = object_detection_working_directory / 'predictions'
    directory_to_save_predictions.mkdir(exist_ok=True)
    
    detection_matrix = []
    
    print('Loading detection model ...')

    detection_model = load_saved_model_for_inference(str(path_to_saved_model))

    category_index, label_map_dict = load_label_map_info(str(label_map_path))
    
    
    for subdirectory_to_detect in directory_with_subdirectories_to_detect.iterdir():
        
        name_of_subdirectory_to_detect = subdirectory_to_detect.name
        
        directory_containing_images_to_detect = subdirectory_to_detect / 'parent_images'

        image_file_paths = directory_containing_images_to_detect.iterdir()

        directory_to_save_detection_figures = directory_to_save_predictions / name_of_subdirectory_to_detect

        shutil.rmtree(directory_to_save_detection_figures, ignore_errors=True)
        directory_to_save_detection_figures.mkdir()

        score_threshold = 0.3

        start_time = time.time()

        for image_file_path in image_file_paths:

            figname = image_file_path.stem

            print(f'Detecting fauna in image {figname} ...')

            image_tensor = read_image_into_a_tensor(image_file_path)

            detections = detect_objects_in_image(detection_model, image_tensor)

            number_of_plausible_detections = np.count_nonzero(detections['detection_scores'].numpy() > score_threshold)

            if number_of_plausible_detections > 0:

                selected_detection_matrix = generate_matrix_of_detections(detections, score_threshold, image_file_path)

                detection_matrix.append(selected_detection_matrix)

                visualize_detections(image_tensor, detections, category_index,score_threshold, directory_to_save_detection_figures,figname)

        end_time = time.time()

        time_taken = time.gmtime(end_time - start_time)

        with open(directory_to_save_detection_figures/'processing_time.txt', 'w') as file:

            print(f'Finished making detections in {time_taken.tm_hour} hours, {time_taken.tm_min} minutes and {time_taken.tm_sec} seconds.', file=file)
    
    complete_detection_matrix = np.concatenate(detection_matrix, axis=0)
    
    detection_results_csv_file_name = directory_to_save_predictions / 'detections_summary_table.csv'

    save_georeferenced_detection_results_as_csv(complete_detection_matrix, label_map_path, detection_results_csv_file_name)