from pathlib import Path
import pandas as pd
import numpy as np
import requests

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
    
    return


def generate_YOLO_object_classes_text_file(path_to_YOLO_input_datasheet, path_to_YOLO_object_classes_text_file):
    '''
    Parse the YOLO datasheet and generate the object classes text file
    
    '''
    path_to_YOLO_input_datasheet = Path(path_to_YOLO_input_datasheet)
    
    YOLO_input_datasheet_df = pd.read_csv(path_to_YOLO_input_datasheet).sort_values(by='object_index')
    
    ordered_labels = YOLO_input_datasheet_df.groupby('object_index').object_name.first()
    
    ordered_labels_as_strings = '\n'.join(ordered_labels)
    
    with open(path_to_YOLO_object_classes_text_file,'w') as file:
        
        file.write(ordered_labels_as_strings)
        
    return

def generate_YOLO_anchors_and_mask_text_files(path_to_YOLO_anchors_file, path_to_YOLO_mask_file):
    '''
    Generate the anchor file as defined in the source code
    
    '''
    anchors_column_1 = [10,16,33,30,62,59,116,156,373]
    anchors_column_2 = [13,30,23,61,45,119,90,198,326]
    
    masks_column_1 = [6,3,0]
    masks_column_2 = [7,4,1]
    masks_column_3 = [8,5,2]
    
    anchors = pd.DataFrame({'col_1':anchors_column_1, 'col_2':anchors_column_2})
    masks = pd.DataFrame({'col_1':masks_column_1, 'col_2':masks_column_2, 'col_3':masks_column_3})
    
    anchors.to_csv(path_to_YOLO_anchors_file, header=False, index=False)
    masks.to_csv(path_to_YOLO_mask_file, header=False, index=False)
    
    return


def download_the_darknet_config_file(darknet_config_url, path_to_YOLO_darknet_config_file):
    '''
    Download weights file from darknet
    
    '''
    r = requests.get(darknet_config_url)
    
    darknet_file_contents = r.text
    
    with open(path_to_YOLO_darknet_config_file, 'w') as file:
        
        file.write(darknet_file_contents)
    
        
        
        
    
        