from pathlib import Path

########################## Segmentation, VAE training and outlier detection #########################
DIVE_TO_PROCESS = 'test_dive'

UNSUPERVISED_LEARNING_DIR = Path.cwd().parent / 'data/unsupervised_outlier_detection/' #Relative to src directory

DIVE_WORKING_DIR = UNSUPERVISED_LEARNING_DIR / f'{DIVE_TO_PROCESS}'
    
DIVE_PARENT_IMAGES_DIR = DIVE_WORKING_DIR / 'parent_images'

DIVE_SAMPLED_BACKGROUND_IMAGES_DIR = DIVE_WORKING_DIR / 'background_images'

DIVE_PICKLED_ITEMS_DIR = DIVE_WORKING_DIR / 'pickled_items'

DIVE_OUTPUT_DIR = DIVE_WORKING_DIR / 'detection_outputs'

NUMBER_OF_IMAGES_TO_SAMPLE_AS_BACKGROUND = 10

BATCH_SIZE = 32

LATENT_DIMENSION = 100

TRAINING_EPOCHS = 5

CONTAMINATION = 'auto'
#####################################################################################################
