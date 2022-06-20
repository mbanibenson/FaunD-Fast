from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from itertools import chain

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
    set_of_annotations = []
    
    with open(text_file_path) as file:
        
        for line in file:
            
            annotations = line.split()
        
            object_index = int(annotations[0])

            object_x_center = float(annotations[1])

            object_y_center = float(annotations[2])

            object_x_width = float(annotations[3])

            object_y_height = float(annotations[4])

            min_y = object_y_center - (0.5*object_y_height)

            max_y = object_y_center + (0.5*object_y_height)

            min_x = object_x_center - (0.5*object_x_width)

            max_x = object_x_center + (0.5*object_x_width)
            
            image_name = text_file_path.with_suffix('.JPG').name
            
            set_of_annotations.append((object_index, min_y, min_x, max_y, max_x, image_name))
    
    return set_of_annotations
        
        


# def generate_object_detection_validation_datasheet_from_manual_annotations(directory_with_manual_annotations_text_files, directory_with_validation_images, path_to_object_detection_validation_datasheet):
#     '''
#     Process manual annotations into a csv file to be consumed by tfrecords creation utility
    
#     '''
#     directory_with_manual_annotations_text_files = Path(directory_with_manual_annotations_text_files)
    
#     directory_with_validation_images = Path(directory_with_validation_images)
    
#     path_to_object_detection_validation_datasheet = Path(path_to_object_detection_validation_datasheet)
    
#     validation_images_file_paths = list(directory_with_validation_images.iterdir())
    
#     validation_images_file_names = [fp.name for fp in validation_images_file_paths]
    
#     validation_images_annotations_text_file_names = [fp.with_suffix('.txt').name for fp in validation_images_file_paths]
    
#     validation_images_annotations_text_file_paths = [(directory_with_manual_annotations_text_files / f_name) for f_name in validation_images_annotations_text_file_names]
    
#     parsed_annotations = [parse_manually_annotated_text_file_in_yolo_format_from_labelimg(txt_file) for txt_file in validation_images_annotations_text_file_paths]
    
#     validation_dataframe = pd.DataFrame(parsed_annotations, columns = ('object_index', 'min_y', 'min_x', 'max_y', 'max_x'))
    
#     validation_dataframe['image_path'] = validation_images_file_paths
    
#     validation_dataframe['image_name'] = validation_images_file_names
    
#     label_map_path = directory_with_manual_annotations_text_files / 'classes.txt'
    
#     with open(label_map_path) as file:
        
#         label_map = [label.rstrip() for label in file.readlines()]        
        
#     validation_dataframe['object_name'] = [label_map[i] for i in validation_dataframe.object_index.tolist()]
    
#     validation_dataframe['object_index'] = validation_dataframe['object_index'] + 1
    
#     validation_dataframe.to_csv(path_to_object_detection_validation_datasheet, index=False)
    
#     return

def generate_object_detection_training_and_validation_datasheet_from_manual_annotations(directory_with_manual_annotations_text_files, directory_with_images, path_to_object_detection_training_datasheet, path_to_object_detection_validation_datasheet):
    '''
    Process manual annotations into a csv file to be consumed by tfrecords creation utility
    
    '''
    directory_with_manual_annotations_text_files = Path(directory_with_manual_annotations_text_files)
    
    path_to_object_detection_training_datasheet = Path(path_to_object_detection_training_datasheet)
    
    path_to_object_detection_validation_datasheet = Path(path_to_object_detection_validation_datasheet)
    
    directory_with_images = Path(directory_with_images)

    annotations_text_file_paths = list(directory_with_manual_annotations_text_files.glob('SO268*.txt'))
    
    empty_annotations = [fp for fp in annotations_text_file_paths if not fp.read_text()]
    
    print(f'Found {len(annotations_text_file_paths)} annotations with {len(empty_annotations)} empty annotation files...')
    
    [fp.unlink() for fp in empty_annotations]
    
    parsed_annotations_nested = [parse_manually_annotated_text_file_in_yolo_format_from_labelimg(txt_file) for txt_file in annotations_text_file_paths if txt_file not in empty_annotations]
    
    parsed_annotations = chain.from_iterable(parsed_annotations_nested)
    
    annotations_dataframe = pd.DataFrame(parsed_annotations, columns = ('object_index', 'min_y', 'min_x', 'max_y', 'max_x', 'image_name'))
    
    #annotations_dataframe['image_name'] = [fp.with_suffix('.JPG').name for fp in annotations_text_file_paths]
    
    annotations_dataframe['image_path'] = [(directory_with_images / im_name) for im_name in annotations_dataframe.image_name.tolist()]
    
    label_map_path = directory_with_manual_annotations_text_files / 'classes.txt'
    
    with open(label_map_path) as file:
        
        label_map = [label.rstrip() for label in file.readlines()]        
        
    annotations_dataframe['object_name'] = [label_map[i] for i in annotations_dataframe.object_index.tolist()]
        
    annotations_dataframe = annotations_dataframe.loc[annotations_dataframe.object_name.ne('Fauna')]
    
    class_names = annotations_dataframe['object_name']
    
    le = LabelEncoder()
    
    le.fit(class_names)
    
    annotations_dataframe['object_index'] = le.transform(class_names) + 1
    
    #annotations_dataframe = annotations_dataframe.loc[annotations_dataframe.object_name.ne('Coral')]
    
    #annotations_dataframe['object_index'] = annotations_dataframe['object_index'] + 1
    
    #indices = annotations_dataframe.index.tolist()
    image_names = annotations_dataframe.image_name.unique()
    
    training_image_names, test_image_names = train_test_split(image_names, test_size=0.1, random_state=100)
    
    training_annotations_df = annotations_dataframe.loc[annotations_dataframe.image_name.isin(training_image_names)]
    
    validation_annotations_df = annotations_dataframe.loc[annotations_dataframe.image_name.isin(test_image_names)]
    
    print(f'Saving datasheets with {len(training_annotations_df)} training and {len(validation_annotations_df)} validation examples ...')
    
    #training_annotations_df.to_csv(path_to_object_detection_training_datasheet, index=False)
    
    training_annotations_df.to_csv(path_to_object_detection_training_datasheet, index=False)
    
    validation_annotations_df.to_csv(path_to_object_detection_validation_datasheet, index=False)
    
    return le

def generate_label_map_from_lebelencoder(trained_label_encoder, path_to_label_map):
    '''
    Infer classes from a trained label encoder and write to label map
    
    '''
    classes = trained_label_encoder.classes_
    
    with open(path_to_label_map, 'w') as file:
        
        for index, label in enumerate(classes, start=1): 
            
            label_template = f'item {{\n\n id: {index}\n\n name: "{label}"\n\n}}\n\n'
        
            file.write(label_template)
            

def generate_label_map_from_labelimg_annotations(directory_with_labelimg_annotations, path_to_label_map):
    '''
    Process labelimg annotations to generate label map
    
    '''
    labelimg_label_map_path = directory_with_labelimg_annotations / 'classes.txt'
    
    with open(labelimg_label_map_path) as file:
        
        labelimg_label_map = [label.rstrip() for label in file.readlines()]
        
        labelimg_label_map.remove('Fauna')
        
        
    with open(path_to_label_map, 'w') as file:
        
        for index, label in enumerate(labelimg_label_map, start=1): 
            
            label_template = f'item {{\n\n id: {index}\n\n name: "{label}"\n\n}}\n\n'
        
            file.write(label_template)

def generate_label_map_from_object_detection_input_datasheet(path_to_object_detection_training_datasheet, path_to_object_detection_validation_datasheet, path_to_label_map):
    '''
    Generate label map from the generated datasheet
    
    '''
    path_to_object_detection_training_datasheet = Path(path_to_object_detection_training_datasheet)
    
    path_to_object_detection_validation_datasheet = Path(path_to_object_detection_validation_datasheet)
    
    path_to_label_map = Path(path_to_label_map)
    
    train_df = pd.read_csv(path_to_object_detection_training_datasheet)#.sort_values(by='object_index')
    
    val_df = pd.read_csv(path_to_object_detection_validation_datasheet)#.sort_values(by='object_index')
    
    input_datasheet_df = pd.concat([train_df, val_df]).sort_values(by='object_index')
    
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
    
    # path_to_object_detection_input_datasheet = object_detection_data_directory / 'object_detection_input_datasheet.csv'
    
    path_to_label_map = object_detection_data_directory / 'SO268_label_map.pbtxt'
    
    path_to_annotations_table = image_viewer_directory / 'annotations.csv'
    
    directory_with_manual_annotations_text_files = image_viewer_directory / 'yolo_annotations_labelimg'
    
    directory_with_images = image_viewer_directory / 'parent_images'
    
    path_to_object_detection_training_datasheet = object_detection_data_directory / 'object_detection_input_datasheet.csv'
    
    path_to_object_detection_validation_datasheet = object_detection_data_directory / 'object_detection_validation_datasheet.csv'
    
    trained_label_encoder = generate_object_detection_training_and_validation_datasheet_from_manual_annotations(directory_with_manual_annotations_text_files, directory_with_images, path_to_object_detection_training_datasheet, path_to_object_detection_validation_datasheet)
    
    generate_label_map_from_lebelencoder(trained_label_encoder, path_to_label_map)
    
    #generate_label_map_from_object_detection_input_datasheet(path_to_object_detection_training_datasheet, path_to_object_detection_validation_datasheet, path_to_label_map)
    
    #generate_label_map_from_labelimg_annotations(directory_with_manual_annotations_text_files, path_to_label_map)
    
    # generate_label_map_from_object_detection_input_datasheet(path_to_object_detection_training_datasheet,path_to_object_detection_validation_datasheet, path_to_label_map)
    
#     generate_object_detection_input_datasheet_from_detection_summary_table(path_to_post_processed_summary_table, path_to_object_detection_input_datasheet, path_to_annotations_table)
    
#     generate_object_detection_validation_datasheet_from_manual_annotations(directory_with_manual_annotations_text_files, directory_with_validation_images, path_to_object_detection_validation_datasheet)
    
    