from models.VAE_based_outlier_detection import segment_images_and_return_segments_as_list_of_ndarrays
from parameters import deepsea_fauna_detection_params
import pickle

if __name__ == '__main__':
    
    directory_containing_images_to_segment = deepsea_fauna_detection_params.DIVE_SAMPLED_BACKGROUND_IMAGES_DIR
    
    directory_containing_pickled_items = deepsea_fauna_detection_params.DIVE_PICKLED_ITEMS_DIR
    
    directory_containing_pickled_items.mkdir(exist_ok=True)

    print('Segmenting background images ...')
    file_paths = list(directory_containing_images_to_segment.rglob('*.JPG'))
    
    list_of_training_patches, training_patch_names, training_patch_bboxes, training_patch_class_labels, list_of_segmented_images, list_of_original_images = segment_images_and_return_segments_as_list_of_ndarrays(file_paths)
    
    with open(directory_containing_pickled_items / f'list_of_training_patches.pickle', 'wb') as f:
        
        pickle.dump(list_of_training_patches, f, pickle.HIGHEST_PROTOCOL)
        
        
    with open(directory_containing_pickled_items / f'list_of_segmented_images.pickle', 'wb') as f:
        
        pickle.dump(list_of_segmented_images, f, pickle.HIGHEST_PROTOCOL)
        
        
    with open(directory_containing_pickled_items / f'list_of_original_images.pickle', 'wb') as f:
        
        pickle.dump(list_of_original_images, f, pickle.HIGHEST_PROTOCOL)
    
    print('Finished segmenting background images ...')