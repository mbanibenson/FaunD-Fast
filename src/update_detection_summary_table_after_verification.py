import pandas as pd
from parameters import deepsea_fauna_detection_params
import shutil
from pathlib import Path


def remove_sort_id_from_patch_name(path_to_patch):
    '''
    Remove sort id from start of patch name so it can be compared to patch name as saved in the summary table
    
    The sort id was only intended fr sorting the patches on disk based on conficence scores
    
    '''
    
    return path_to_patch.stem.split('_', maxsplit=1)[1]


def update_detections_table(path_to_directory_with_cropped_patches, path_to_original_detections_summary_table, path_to_updated_detections_summary_table):
    '''
    Update table after detected patches have been verified and e.g. wrong detections deleted
    
    '''
    
    original_megafauna_df = pd.read_csv(path_to_original_detections_summary_table)
    
    current_patch_ids_after_verification = pd.DataFrame({'patch_id':[remove_sort_id_from_patch_name(path_to_patch) for path_to_patch in Path(path_to_directory_with_cropped_patches).rglob('*.png')]}) #Remove the sort id
    
    updated_megafauna_df = pd.merge(original_megafauna_df, current_patch_ids_after_verification, on='patch_id', how='inner')
    
    updated_megafauna_df.to_csv(path_to_updated_detections_summary_table, index=False)
    
    return updated_megafauna_df


if __name__ == '__main__':
    
    object_detection_directory = deepsea_fauna_detection_params.OBJECT_DETECTION_DIR
    
    path_to_directory_with_cropped_patches = deepsea_fauna_detection_params.DETECTION_PATCHES_DIRECTORY
    
    path_to_original_detections_summary_table = object_detection_directory / 'faster_rcnn_with_detection_checkpoint/predictions/detections_summary_table.csv'
    
    path_to_updated_detections_summary_table = path_to_directory_with_cropped_patches / 'updated_detections_summary_table.csv'
    
    print('Updating detections table ...')
    
    updated_megafauna_df = update_detections_table(path_to_directory_with_cropped_patches, path_to_original_detections_summary_table, path_to_updated_detections_summary_table)
    
    print(f'Done ({len(updated_megafauna_df)} entries)')