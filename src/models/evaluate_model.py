from pathlib import Path
import pandas as pd

def extract_time_from_image_file_name(series_with_file_names):
    '''
    Given a file path, extract its acquisition time
    
    '''
    def split_file_name_to_extract_time_components(image_name):
        #Extract the time from the file name
        image_name_splits = image_name.split('_')

        image_time = f'{image_name_splits[-2]}_{image_name_splits[-1]}'

        return image_time
    
    series_with_acquisition_time_as_string = series_with_file_names.map(extract_time_from_image_file_name)
    
    series_with_acquisition_time_as_date_format = pd.to_datetime(series_with_acquisition_time_as_string, format='%Y%m%d_%H%M%S')
    
    return series_with_acquisition_time_as_date_format


def count_detections_per_image(directory_containing_detections):
    '''
    
    Tabulate the number of superpixels flagged as positive detections.
    
    '''
    directory_containing_detections = Path(directory_containing_detections)
    
    tables_summarizing_detections = []
    
    for directory in directory_containing_detections.iterdir():
        
        if (directory.is_dir() and directory.name.startswith('SO268')):
            
            detections_file_name = directory/'detections_summary_table.csv'
            
            table_summarizing_detections = pd.read_csv(detections_file_name)
            
            tables_summarizing_detections.append(table_summarizing_detections)
            
    master_table_summarizing_detections = pd.concat(tables_summarizing_detections)
    
    master_table_with_counts_per_image = master_table_summarizing_detections.groupby('parent_image_name').size().reset_index()
    
    master_table_with_counts_per_image['dive'] = master_table_with_counts_per_image['parent_image_name'].map(lambda x: x.split('_')[1])
    
    
    master_table_with_counts_per_image['acquisition_time'] = extract_time_from_image_file_name(master_table_with_counts_per_image['parent_image_name'])
    
    master_table_with_counts_per_image.to_csv(directory_containing_detections/'table_with_detection_counts_per_image.csv', index=True)
    
    return

if __name__ == '__main__':
    
    directory_containing_detections = '/home/mbani/mardata/datasets/positively_detected_fauna_experimental'
    
    count_detections_per_image(directory_containing_detections)
    
    
    
    
    
            
            
    