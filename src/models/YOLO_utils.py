from pathlib import Path
import pandas as pd
import numpy as np

def convert_segmentation_bbox_to_yolo_format(bbox_column_as_pandas_series):
    '''
    Convert bounding box column to a dataframe with YOLO type coordinates
    
    '''
    split_dataframe = pd.DataFrame(bbox_column_as_pandas_series.str.split('-').tolist(),columns=['y0', 'x0', 'y1', 'x1']).loc[:,['x0', 'y0', 'x1', 'y1']]
    
    desired_dataframe = (split_dataframe.astype('float') / [1680, 1120, 1680, 1120]) #Convert to normalized coordinates
    
    return desired_dataframe


def generate_object_label_columns():
    '''
    Generate object name and object index columns. Also generate the label map
    
    '''
    pass


def generate_YOLO_input_datasheet_from_detection_summary_table(path_to_post_processed_summary_table, path_to_YOLO_input_datasheet):
    '''
    Reformat the post processed datasheet to generate the corresponding YOLO compatible version
    
    '''
    path_to_post_processed_summary_table = Path(path_to_post_processed_summary_table)
    
    post_processed_summary_table = pd.read_csv(path_to_post_processed_summary_table)
    
    bbox_column = post_processed_summary_table.bbox
    
    image_column = post_processed_summary_table.parent_image_url.tolist()
    
    object_name_column = ['fauna'] * len(image_column)
    
    object_index_column = [0] * len(image_column)
    
    yolo_labels_dataframe = pd.DataFrame({'image':image_column, 'object_name':object_name_column, 'object_index':object_index_column})
    
    yolo_coordinates_dataframe = convert_segmentation_bbox_to_yolo_format(bbox_column)
    
    YOLO_input_datasheet = pd.concat([yolo_labels_dataframe, yolo_coordinates_dataframe], axis=1)
    
    YOLO_input_datasheet.to_csv(path_to_YOLO_input_datasheet, index=False)