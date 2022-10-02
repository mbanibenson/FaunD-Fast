import pandas as pd
from parameters import deepsea_fauna_detection_params
import shutil
from pathlib import Path


def update_detections_table(path_to_directory_with_cropped_patches, path_to_original_detections_summary_table, path_to_updated_detections_summary_table):
    '''
    Update table after detected patches have been verified and e.g. wrong detections deleted
    
    '''
    
    original_megafauna_df = pd.read_csv(path_to_original_detections_summary_table)
    
    current_patch_ids_after_verification = pd.DataFrame[{'patch_id':file_path.stem for file_path in Path(path_to_directory_with_cropped_patches).rglob('.png')}]
    
    updated_megafauna_df = pd.merge(original_megafauna_df, current_patch_ids_after_verification, on='patch_id', how='inner')
    
    updated_megafauna_df.to_csv(path_to_updated_detections_summary_table, index=False)
    
    return


if __name__ == '__main__':
    
    path_to_directory_with_cropped_patches = #
    
    path_to_original_detections_summary_table = object_detection_directory / 'faster_rcnn_with_detection_checkpoint/predictions/detections_summary_table.csv'
    
    path_to_updated_detections_summary_table = #
    
    print('Updating detections table ...')
    
    update_detections_table(path_to_directory_with_cropped_patches, path_to_original_detections_summary_table, path_to_updated_detections_summary_table)
    
    print('Done.')