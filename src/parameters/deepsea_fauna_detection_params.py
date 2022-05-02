from pathlib import Path

########################## Segmentation, VAE training and outlier detection #########################
DIVE_TO_PROCESS = 'dive_126'

UNSUPERVISED_LEARNING_DIR = Path.cwd().parents[1] / 'data/unsupervised' #Dont modify

DIVE_WORKING_DIR = UNSUPERVISED_LEARNING_DIR / f'{dive_to_process}'
    
DIVE_PARENT_IMAGES_DIR = DIVE_WORKING_DIR / 'parent_images'

DIVE_SAMPLED_BACKGROUND_IMAGES_DIR = DIVE_WORKING_DIR / 'background_images'

DIVE_PICKLED_ITEMS_DIR = DIVE_WORKING_DIR / 'pickled_items'

DIVE_OUTPUT_DIR = DIVE_WORKING_DIR / 'detection_outputs'

BATCH_SIZE = 32

LATENT_DIMENSION = 100

TRAINING_EPOCHS = 20

CONTAMINATION = 'auto'
#####################################################################################################
