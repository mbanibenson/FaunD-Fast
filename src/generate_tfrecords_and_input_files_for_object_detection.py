from pathlib import Path
from custom_object_detection.tf_object_detection_utilities import create_train_val_input_tfrecords
from custom_object_detection.tf_object_detection_utilities import download_checkpoint_for_pretraining

if __name__ == '__main__':
    
    working_directory = Path.cwd().parents[0]
    
    object_detection_directory = working_directory / 'fauna_detection_with_tensorflow_object_detection_api'
    
    object_detection_data_directory = object_detection_directory / 'data'
    
    path_to_train_csv_with_labels = object_detection_data_directory / 'object_detection_input_datasheet.csv'
    
    path_to_val_csv_with_labels = object_detection_data_directory / 'object_detection_validation_datasheet.csv'
    
    path_to_label_map = object_detection_data_directory / 'SO268_label_map.pbtxt'
    
    path_to_train_tfrecord_file = object_detection_data_directory / 'train.tfrecord'
    
    path_to_validation_tfrecord_file = object_detection_data_directory / 'validation.tfrecord'
    
    directory_to_save_config_file = object_detection_directory / 'my_model_dir/'
        
    detection_checkpoint_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8.tar.gz'
    
    directory_to_save_checkpoint = object_detection_directory / 'my_model_dir/'

    create_train_val_input_tfrecords(path_to_train_csv_with_labels, path_to_val_csv_with_labels, path_to_label_map, path_to_train_tfrecord_file, path_to_validation_tfrecord_file)
    
    download_checkpoint_for_pretraining(detection_checkpoint_url, directory_to_save_checkpoint)

    
    
    