from models.YOLO_utils import generate_YOLO_input_datasheet_from_detection_summary_table
from models.YOLO_utils import generate_YOLO_object_classes_text_file
from models.YOLO_utils import generate_YOLO_anchors_and_mask_text_files
from models.YOLO_utils import download_the_darknet_config_file
from pathlib import Path

if __name__ == '__main__':
    
    data_directory = Path.cwd().parents[0] / 'data'

    image_viewer_directory = Path.cwd().parents[0] / 'reports/mbani-image-viewer'
    
    YOLO_directory = data_directory / 'YOLO'

    path_to_post_processed_summary_table = image_viewer_directory / 'master_detections_summary_table.csv'
    
    path_to_YOLO_input_datasheet = YOLO_directory / 'YOLO_input_datasheet.csv'
    
    path_to_YOLO_object_classes_text_file = YOLO_directory / 'YOLO_object_classes_text_file.txt'
    
    path_to_YOLO_anchors_file = YOLO_directory / 'YOLO_anchors_file.txt'
    
    path_to_YOLO_mask_file = YOLO_directory / 'YOLO_mask_file.txt'
    
    darknet_config_url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg'
    
    path_to_YOLO_darknet_config_file = YOLO_directory / 'YOLO_darknet_config.cfg'
    
    generate_YOLO_input_datasheet_from_detection_summary_table(path_to_post_processed_summary_table, path_to_YOLO_input_datasheet)
    
    generate_YOLO_object_classes_text_file(path_to_YOLO_input_datasheet, path_to_YOLO_object_classes_text_file)
    
    generate_YOLO_anchors_and_mask_text_files(path_to_YOLO_anchors_file, path_to_YOLO_mask_file)
    
    download_the_darknet_config_file(darknet_config_url, path_to_YOLO_darknet_config_file)