from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def convert_segmentation_bbox_to_normalized_format(bbox_column_as_pandas_series):
    '''
    Convert bounding box column to a normalized format compatible with tf object detection api
    
    '''
    split_dataframe = pd.DataFrame(bbox_column_as_pandas_series.str.split('-').tolist(),columns=['min_y', 'min_x', 'max_y', 'max_x'])
    
    desired_dataframe = (split_dataframe.astype('float') / [1120, 1680, 1120, 1680]) #Convert to normalized coordinates
    
    return desired_dataframe


def generate_object_detection_input_datasheet_from_detection_summary_table(path_to_post_processed_summary_table, path_to_object_detection_input_datasheet, path_to_annotations_table):
    '''
    Reformat the post processed datasheet to generate the corresponding YOLO compatible version
    
    '''
    path_to_post_processed_summary_table = Path(path_to_post_processed_summary_table)
    
    post_processed_summary_table = pd.read_csv(path_to_post_processed_summary_table)
    
    
    path_to_annotations_table = Path(path_to_annotations_table)
    
    annotations_table = pd.read_csv(path_to_annotations_table, delimiter=';')
    
    annotated_summary_table = pd.merge(post_processed_summary_table, annotations_table, on='patch_name')
    
    annotated_summary_table = annotated_summary_table.loc[annotated_summary_table.classification.ne('Nothing')]
    
    
    bbox_column = annotated_summary_table.bbox
    
    image_path_column = annotated_summary_table.parent_image_url.tolist()
    
    image_name_column = [Path(fp).name for fp in image_path_column]
    
    object_name_column = annotated_summary_table.classification.tolist()
    
    label_encoder = LabelEncoder()
    
    encoded_labels = label_encoder.fit_transform(object_name_column)
    
    object_index_column = (encoded_labels + 1).tolist()
    
    object_detection_labels_dataframe = pd.DataFrame({'image_path':image_path_column, 'image_name':image_name_column, 'object_name':object_name_column, 'object_index':object_index_column})
    
    object_detection_coordinates_dataframe = convert_segmentation_bbox_to_normalized_format(bbox_column)
    
    object_detection_input_datasheet = pd.concat([object_detection_labels_dataframe, object_detection_coordinates_dataframe], axis=1)
    
    object_detection_input_datasheet.to_csv(path_to_object_detection_input_datasheet, index=False)
    
    return

def parse_manually_annotated_text_file_in_yolo_format_from_labelimg(text_file_path):
    '''
    Parse a yolo type annotation file
    
    '''
    with open(text_file_path) as file:
        
        annotations = file.readline().split()
        
    object_index = int(annotations[0])
    
    object_x_center = float(annotations[1])
    
    object_y_center = float(annotations[2])
    
    object_x_width = float(annotations[3])
    
    object_y_height = float(annotations[4])
    
    min_y = object_y_center - (0.5*object_y_height)
    
    max_y = object_y_center + (0.5*object_y_height)
    
    min_x = object_x_center - (0.5*object_x_width)
    
    max_x = object_x_center + (0.5*object_x_width)
    
    return object_index, min_y, min_x, max_y, max_x
        
        


def generate_object_detection_validation_datasheet_from_manual_annotations(directory_with_manual_annotations_text_files, directory_with_validation_images, path_to_object_detection_validation_datasheet):
    '''
    Process manual annotations into a csv file to be consumed by tfrecords creation utility
    
    '''
    directory_with_manual_annotations_text_files = Path(directory_with_manual_annotations_text_files)
    
    directory_with_validation_images = Path(directory_with_validation_images)
    
    path_to_object_detection_validation_datasheet = Path(path_to_object_detection_validation_datasheet)
    
    validation_images_file_paths = list(directory_with_validation_images.iterdir())
    
    validation_images_file_names = [fp.name for fp in validation_images_file_paths]
    
    validation_images_annotations_text_file_names = [fp.with_suffix('.txt').name for fp in validation_images_file_paths]
    
    validation_images_annotations_text_file_paths = [(directory_with_manual_annotations_text_files / f_name) for f_name in validation_images_annotations_text_file_names]
    
    parsed_annotations = [parse_manually_annotated_text_file_in_yolo_format_from_labelimg(txt_file) for txt_file in validation_images_annotations_text_file_paths]
    
    validation_dataframe = pd.DataFrame(parsed_annotations, columns = ('object_index', 'min_y', 'min_x', 'max_y', 'max_x'))
    
    validation_dataframe['image_path'] = validation_images_file_paths
    
    validation_dataframe['image_name'] = validation_images_file_names
    
    label_map_path = directory_with_manual_annotations_text_files / 'classes.txt'
    
    with open(label_map_path) as file:
        
        label_map = [label.rstrip() for label in file.readlines()]        
        
    validation_dataframe['object_name'] = [label_map[i] for i in validation_dataframe.object_index.tolist()]
    
    validation_dataframe['object_index'] = validation_dataframe['object_index'] + 1
    
    validation_dataframe.to_csv(path_to_object_detection_validation_datasheet, index=False)
    
    return
    
    
        
    


def generate_label_map_from_object_detection_input_datasheet(path_to_object_detection_input_datasheet, path_to_label_map):
    '''
    Generate label map from the generated datasheet
    
    '''
    path_to_object_detection_input_datasheet = Path(path_to_object_detection_input_datasheet)
    
    path_to_label_map = Path(path_to_label_map)
    
    input_datasheet_df = pd.read_csv(path_to_object_detection_input_datasheet).sort_values(by='object_index')
    
    string_labels = input_datasheet_df.groupby('object_index').object_name.first()
    
    with open(path_to_label_map, 'w') as file:
        
        for index, label in enumerate(string_labels, start=1): 
            
            label_template = f'item {{\n\n id: {index}\n\n name: "{label}"\n\n}}\n\n'
        
            file.write(label_template)
            
            
            

if __name__ == '__main__':
    
    working_directory = Path.cwd().parents[0]
    
    image_viewer_directory = working_directory / 'custom_annotation_tool'
    
    object_detection_directory = working_directory / 'fauna_detection_with_tensorflow_object_detection_api'
    
    object_detection_data_directory = object_detection_directory / 'data'
    
    object_detection_data_directory.mkdir(exist_ok=True)

    path_to_post_processed_summary_table = image_viewer_directory / 'master_detections_summary_table.csv'
    
    path_to_object_detection_input_datasheet = object_detection_data_directory / 'object_detection_input_datasheet.csv'
    
    path_to_label_map = object_detection_data_directory / 'SO268_label_map.pbtxt'
    
    path_to_annotations_table = image_viewer_directory / 'annotations.csv'
    
    directory_with_manual_annotations_text_files = object_detection_directory / 'performance_assessment/manual_annotations'
    
    directory_with_validation_images = object_detection_directory / 'performance_assessment/validation_images'
    
    path_to_object_detection_validation_datasheet = object_detection_data_directory / 'object_detection_validation_datasheet.csv'
    
    generate_object_detection_input_datasheet_from_detection_summary_table(path_to_post_processed_summary_table, path_to_object_detection_input_datasheet, path_to_annotations_table)
    
    generate_object_detection_validation_datasheet_from_manual_annotations(directory_with_manual_annotations_text_files, directory_with_validation_images, path_to_object_detection_validation_datasheet)
    
    generate_label_map_from_object_detection_input_datasheet(path_to_object_detection_input_datasheet, path_to_label_map)