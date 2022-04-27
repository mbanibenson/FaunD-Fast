from custom_object_detection.tf_object_detection_utilities import load_saved_model_for_inference
from custom_object_detection.tf_object_detection_utilities import detect_objects_in_image
from pathlib import Path

if __name__ == '__main__':
    
    detection_model, category_index = load_saved_model_for_inference(path_to_saved_model, path_to_label_map)
    
    detections = detect_objects_in_image(detection_model, image_file_path)