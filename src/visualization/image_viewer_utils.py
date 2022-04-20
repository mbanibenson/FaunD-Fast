import numpy as np
import time
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import shutil



def create_post_processed_detections_summary_table(directory_containing_fauna_patches, directory_containing_unsupervised_outlier_detection_results, path_to_post_processed_summary_table):
    '''
    Create a new table that has records for the post processed patches - after sorting by CNN
    
    ##TODO: Loop through dives directory and merge the detection_summary_tables.csv
    
    ##TODO: Merge the master detection summary table with the one containing georeferenced coordinates
    
    '''
    directory_containing_fauna_patches = Path(directory_containing_fauna_patches)
    
    directory_containing_unsupervised_outlier_detection_results = Path(directory_containing_unsupervised_outlier_detection_results)
    
    path_to_post_processed_summary_table = Path(path_to_post_processed_summary_table)
    
    original_summary_table_dataframes = []
    
    for directory in directory_containing_unsupervised_outlier_detection_results.iterdir():
        
        outlier_detection_output_directory = directory / 'detection_outputs'
        
        if not outlier_detection_output_directory.exists():
            
            continue
            
        path_to_original_summary_table = outlier_detection_output_directory / 'detections_summary_table.csv'
    
        original_summary_table_dataframe = pd.read_csv(path_to_original_summary_table)
        
        original_summary_table_dataframes.append(original_summary_table_dataframe)
        
    master_original_summary_table_dataframe = pd.concat(original_summary_table_dataframes, axis=0)
    
    assert len(master_original_summary_table_dataframe.columns) == len(original_summary_table_dataframes[0].columns), 'Summary tables merge was not properly done. Unexpected output shape'
    
    fauna_patches_file_names = [fp.stem for fp in directory_containing_fauna_patches.iterdir()]
    
    fauna_patches_df = pd.DataFrame({'patch_name':fauna_patches_file_names})
    
    fauna_patches_df_complete = pd.merge(fauna_patches_df, master_original_summary_table_dataframe, how='left', on='patch_name')
    
    fauna_patches_df_complete.to_csv(path_to_post_processed_summary_table, index=False)
    
    return

def copy_files_to_image_viewer_for_annotation(directory_containing_pure_fauna_patches, directory_containing_unsupervised_outlier_detection_results, path_to_post_processed_summary_table, image_viewer_directory):
    '''
    Copy parent images, patches and master summary table to the image viewer for annotation
    
    '''
    parent_images_directory_in_image_viewer = image_viewer_directory / 'parent_images'
    
    patches_directory_in_image_viewer = image_viewer_directory / 'patches'
    
    shutil.rmtree(parent_images_directory_in_image_viewer, ignore_errors=True)
    
    shutil.rmtree(patches_directory_in_image_viewer, ignore_errors=True)
    
    parent_images_directory_in_image_viewer.mkdir()
    
    #patches_directory_in_image_viewer.mkdir()
    
    post_processed_summary_table = pd.read_csv(path_to_post_processed_summary_table)
    
    original_parent_image_paths = set([fp for fp in directory_containing_unsupervised_outlier_detection_results.rglob('*.JPG') if fp.stem in post_processed_summary_table.parent_image_name.values])
    
    print(f'Found {len(original_parent_image_paths)} parent image paths')
    
    with ThreadPoolExecutor() as executor:
        
        [executor.submit(shutil.copy2, file, parent_images_directory_in_image_viewer) for file in original_parent_image_paths]
    
    name_of_fauna_patches_in_image_viewer = image_viewer_directory / 'patches'
    
    shutil.copytree(directory_containing_pure_fauna_patches, name_of_fauna_patches_in_image_viewer)

    shutil.copy2(path_to_post_processed_summary_table, image_viewer_directory)
    
    return