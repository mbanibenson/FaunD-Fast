from models.YOLO_utils import generate_YOLO_input_datasheet_from_detection_summary_table
from pathlib import Path

if __name__ == '__main__':
    
    data_directory = Path.cwd().parents[0] / 'data'

    image_viewer_directory = Path.cwd().parents[0] / 'reports/mbani-image-viewer'

    path_to_post_processed_summary_table = image_viewer_directory / 'master_detections_summary_table.csv'
    
    path_to_YOLO_input_datasheet = data_directory / 'YOLO/YOLO_input_datasheet.csv'
    
    generate_YOLO_input_datasheet_from_detection_summary_table(path_to_post_processed_summary_table, path_to_YOLO_input_datasheet)