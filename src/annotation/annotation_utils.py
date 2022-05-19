import pandas as pd
from pathlib import Path
import shutil


def write_text_file_of_detections(detections_dataframe, text_file_name):
    '''
    Process the detections and save as yolo compatible text files
    
    '''
    object_index = [0] * len(detections_dataframe)
    
    detections_dataframe = detections_dataframe.map(lambda x: x.split('-')).to_frame(name='coords')
    
    with open(text_file_name, 'w') as file:
    
        for row in detections_dataframe.itertuples():

            coords = row.coords
            
            normalizing_factors = [1120, 1680, 1120, 1680]

            coords = [(float(val) / normalizing_factor) for val, normalizing_factor in zip(coords, normalizing_factors)]

            y_min, x_min, y_max, x_max = coords

            x_center = 0.5*(x_min + x_max)

            y_center = 0.5*(y_min + y_max)

            width = x_max - x_min

            height = y_max - y_min
            
            object_index = 0
            
            print(f'{object_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}', file=file)
            

def generate_yolo_annotations_from_post_segmentation_summary_table(path_to_detection_summary_table, directory_to_store_annotations):
    '''
    Parse the detection summary table to generate yolo labels
    
    '''
    path_to_detection_summary_table = Path(path_to_detection_summary_table)
    
    directory_to_store_annotations = Path(directory_to_store_annotations)
    
    shutil.rmtree(directory_to_store_annotations, ignore_errors=True)
    directory_to_store_annotations.mkdir()
    
    detections_df = pd.read_csv(path_to_detection_summary_table)
    
    label_groups = detections_df.groupby('parent_image_name')['bbox']
    
    with open(directory_to_store_annotations / 'classes.txt', 'w') as file:
        
        file.write('fauna\n')
    
    for group_id, group in label_groups:
        
        text_file_name = f'{directory_to_store_annotations}/{group_id}.txt'
        
        group_of_detections_dataframe = group
        
        write_text_file_of_detections(group_of_detections_dataframe, text_file_name)