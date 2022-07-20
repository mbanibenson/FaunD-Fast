import pandas as pd
import tensorflow as tf
from skimage.transform import resize
from skimage.io import imread, imsave
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from parameters import deepsea_fauna_detection_params
import shutil
from skimage import img_as_ubyte
from itertools import accumulate

def crop_detections_for_verification(path_to_detections_summary_table):
    '''
    Crop the detections so they may be visualized
    
    '''
    detected_megafauna_df = pd.read_csv(path_to_detections_summary_table)

    patch_ids = detected_megafauna_df.groupby('parent_image_name')['parent_image_name'].transform(lambda x: accumulate([1]*len(x)))

    detected_megafauna_df['patch_id'] = detected_megafauna_df.parent_image_name.str.cat(patch_ids.astype('str'), sep='#')
    
    morphotype_groupings = detected_megafauna_df.groupby('object_class_name')
    
    cropped_detections_to_display = {}
    
    for morphotype_class, morphotype_group in morphotype_groupings:
        
        morphotype_group_subset = morphotype_group.sort_values(by='score', ascending=False)#.iloc[:20]
        
        cropped_detections = []
        
        for detection in morphotype_group_subset.itertuples():
            
            image_array = imread(detection.parent_image_path)
            
            offset_height = int(detection.ymin)
            
            offset_width = int(detection.xmin)
            
            target_height = int(detection.ymax) - int(detection.ymin)
            
            target_width = int(detection.xmax) - int(detection.xmin)
            
            cropped_detection = tf.image.crop_to_bounding_box(image_array, offset_height, offset_width, target_height, target_width).numpy()
            
            cropped_detection_id = detection.patch_id
            
            cropped_detections.append({'cropped_detection': resize(cropped_detection, (96,96)), 'cropped_detection_id':cropped_detection_id})
            
        cropped_detections_to_display[morphotype_class] = cropped_detections
        
        print(f'{morphotype_class}: {len(cropped_detections)}')
        
    return cropped_detections_to_display


def save_cropped_detections_to_disk(directory_to_save_cropped_detections, cropped_detections_to_display):
    '''
    Save cropped detections to disk
    
    '''
    directory_to_save_cropped_detections = Path(directory_to_save_cropped_detections)
    
    shutil.rmtree(directory_to_save_cropped_detections, ignore_errors=True)
    directory_to_save_cropped_detections.mkdir()
    
    for morphotype_class, patches_and_patch_ids in cropped_detections_to_display.items():
        
        patches = [patch_and_patch_id['cropped_detection'] for patch_and_patch_id in patches_and_patch_ids] 
        
        patch_names = [patch_and_patch_id['cropped_detection_id'] for patch_and_patch_id in patches_and_patch_ids] 
        
        morphotype_directory = directory_to_save_cropped_detections / morphotype_class
        
        morphotype_directory.mkdir()
        
        print (f'Saving {morphotype_class} patches ...')
        
        with ThreadPoolExecutor() as executor:
            
            [executor.submit(imsave, morphotype_directory / f'{sort_id}_{fname}.png', img_as_ubyte(patch)) for sort_id, fname, patch in zip(count(start=1), patch_names, patches)]
            
    return

if __name__ == '__main__':
    
    object_detection_directory = deepsea_fauna_detection_params.OBJECT_DETECTION_DIR
    
    directory_to_save_cropped_detections = deepsea_fauna_detection_params.DETECTION_PATCHES_DIRECTORY
    
    path_to_detections_summary_table = object_detection_directory / 'faster_rcnn_with_detection_checkpoint/predictions/detections_summary_table.csv'
    
    cropped_detections_to_display = crop_detections_for_verification(path_to_detections_summary_table)
    
    save_cropped_detections_to_disk(directory_to_save_cropped_detections, cropped_detections_to_display)
    