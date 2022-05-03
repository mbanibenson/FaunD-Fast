from models.VAE_based_outlier_detection import copy_training_images_from_parent_images
from parameters import deepsea_fauna_detection_params

if __name__ == '__main__':
    
    directory_containing_parent_images = deepsea_fauna_detection_params.DIVE_PARENT_IMAGES_DIR

    directory_containing_sampled_background_images = deepsea_fauna_detection_params.DIVE_SAMPLED_BACKGROUND_IMAGES_DIR
    
    sample_size = deepsea_fauna_detection_params.NUMBER_OF_IMAGES_TO_SAMPLE_AS_BACKGROUND

    print('Copying training images ...')
    
    copy_training_images_from_parent_images(directory_containing_parent_images, directory_containing_sampled_background_images, sample_size)
    
    print('Finished copying training images ...')