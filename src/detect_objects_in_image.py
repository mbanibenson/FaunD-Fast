from custom_object_detection.tf_object_detection_utilities import load_saved_model_for_inference
from custom_object_detection.tf_object_detection_utilities import detect_objects_in_image
from custom_object_detection.tf_object_detection_utilities import load_label_map_info
from custom_object_detection.tf_object_detection_utilities import visualize_detections
from custom_object_detection.tf_object_detection_utilities import read_image_as_ndarray
import matplotlib.pyplot as plt
from pathlib import Path

if __name__ == '__main__':
    
    path_to_saved_model = '/home/mbani/mardata/project-repos/deepsea-fauna-detection/data/tf_object_detection/models/my_model_dir/exported_model_dir/saved_model'
    label_map_path = '/home/mbani/mardata/project-repos/deepsea-fauna-detection/data/tf_object_detection/data/SO268_label_map.pbtxt'
    image_file_path = '/home/mbani/mardata/project-repos/deepsea-fauna-detection/data/unsupervised_outlier_detection/dive_177/parent_images/SO268-2_177-1_OFOS-13_20190510_071413.JPG'
    
    detection_model = load_saved_model_for_inference(path_to_saved_model)

    category_index, label_map_dict = load_label_map_info(label_map_path)
    
    image_tensor = read_image_as_ndarray(image_file_path)
    
    detections = detect_objects_in_image(detection_model, image_tensor)
    
    visualize_detections(image_tensor, detections, category_index)