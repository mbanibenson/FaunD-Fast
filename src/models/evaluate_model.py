from pathlib import Path
import pandas as pd
from zipfile import ZipFile

def extract_time_from_image_file_name(series_with_file_names):
    '''
    Given a file path, extract its acquisition time
    
    '''
    def split_file_name_to_extract_time_components(image_name):
        #Extract the time from the file name
        image_name_splits = image_name.split('_')

        image_time = f'{image_name_splits[-2]}_{image_name_splits[-1]}'

        return image_time
    
    series_with_acquisition_time_as_string = series_with_file_names.map(split_file_name_to_extract_time_components)
    
    series_with_acquisition_time_as_date_format = pd.to_datetime(series_with_acquisition_time_as_string, format='%Y%m%d_%H%M%S')
    
    return series_with_acquisition_time_as_date_format


def count_detections_per_image(directory_containing_detections, directory_to_save_metrics):
    '''
    
    Tabulate the number of superpixels flagged as positive detections.
    
    '''
    directory_containing_detections = Path(directory_containing_detections)
    
    tables_summarizing_detections = []
    
    for directory in directory_containing_detections.iterdir():
        
        if (directory.is_dir() and (directory.name.startswith('SO268') or directory.name.startswith('test'))):
            
            detections_file_name = directory/'detections_summary_table.csv'
            
            table_summarizing_detections = pd.read_csv(detections_file_name)
            
            tables_summarizing_detections.append(table_summarizing_detections)
            
    master_table_summarizing_detections = pd.concat(tables_summarizing_detections)
    
    master_table_with_counts_per_image = master_table_summarizing_detections.groupby('parent_image_name').size().to_frame(name='detection_counts').reset_index()
    
    master_table_with_counts_per_image['dive'] = master_table_with_counts_per_image['parent_image_name'].map(lambda x: x.split('_')[1])
    
    
    master_table_with_counts_per_image['acquisition_time'] = master_table_with_counts_per_image['parent_image_name'].transform(extract_time_from_image_file_name)
    
    master_table_with_counts_per_image.to_csv(directory_to_save_metrics/'table_with_detection_counts_per_image.csv', index=False)
    
    return


def merge_all_detection_summaries_to_master_csv(directory_containing_detections, directory_to_save_master_csv):
    '''
    Gather all the detection summary csvs from the dive by dive detections to one mega csv
    
    '''
    directory_containing_detections = Path(directory_containing_detections)
    
    tables_summarizing_detections = []
    
    for directory in directory_containing_detections.iterdir():
        
        if (directory.is_dir() and (directory.name.startswith('SO268') or directory.name.startswith('test'))):
            
            detections_file_name = directory/'detections_summary_table.csv'
            
            table_summarizing_detections = pd.read_csv(detections_file_name)
            
            tables_summarizing_detections.append(table_summarizing_detections)
            
    master_table_summarizing_detections = pd.concat(tables_summarizing_detections)
    
    georeferenced_master_dataframe_summarizing_detections = append_geospatial_coordinates_to_master_detection_summary_table(master_dataframe_summarizing_detections)
    
    georeferenced_master_dataframe_summarizing_detections.to_csv(directory_to_save_master_csv/'master_detections_summary_table.csv', index=False)
    
    
def append_geospatial_coordinates_to_master_detection_summary_table(master_dataframe_summarizing_detections):
    '''
    Append geospatial coordinates to master dataframe summarizing detections before saving to disk
    
    '''
    path_to_table_with_georeferenced_coords_for_all_photos = Path.cwd().parents[0]/'reports/auxilliary_data/georeferenced_photo_coordinates.zip'
    
    with ZipFile(path_to_table_with_georeferenced_coords_for_all_photos) as myzip:
        
        with myzip.open('georeferenced_photo_coordinates.csv') as myfile:
            
            table_with_georeferenced_coords_for_all_photos = pd.read_csv(myfile)
            
    table_with_georeferenced_coords_for_all_photos['parent_image_name'] = table_with_georeferenced_coords_for_all_photos.Name.map(lambda x: x.split('.')[0])
            
    georeferenced_master_dataframe_summarizing_detections = pd.merge(master_dataframe_summarizing_detections, table_with_georeferenced_coords_for_all_photos, how='left', on='parent_image_name')
    
    return georeferenced_master_dataframe_summarizing_detections
            
            
    
     

# if __name__ == '__main__':
    
#     directory_containing_detections = Path('/home/mbani/mardata/datasets/positively_detected_fauna_experimental')
    
#     directory_to_save_metrics = directory_containing_detections / 'detection_metrics'
    
#     directory_to_save_metrics.mkdir(exist_ok=True)
    
#     count_detections_per_image(directory_containing_detections, directory_to_save_metrics)
    
    
    
    
    
            
            
    