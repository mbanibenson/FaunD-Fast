from pathlib import Path

########################## Segmentation, VAE training and outlier detection #########################
UNSUPERVISED_LEARNING_DIR = Path.cwd().parent / 'data/unsupervised_outlier_detection/' #Relative to src directory

DIVE_TO_PROCESS = 'example_dive'

DIVE_WORKING_DIR = UNSUPERVISED_LEARNING_DIR / f'{DIVE_TO_PROCESS}'
    
DIVE_PARENT_IMAGES_DIR = DIVE_WORKING_DIR / 'parent_images'

DIVE_SAMPLED_BACKGROUND_IMAGES_DIR = DIVE_WORKING_DIR / 'background_images'

DIVE_PICKLED_ITEMS_DIR = DIVE_WORKING_DIR / 'pickled_items'

DIVE_OUTPUT_DIR = DIVE_WORKING_DIR / 'detection_outputs'

NUMBER_OF_IMAGES_TO_SAMPLE_AS_BACKGROUND = 20

BATCH_SIZE = 32

LATENT_DIMENSION = 100

TRAINING_EPOCHS = 50

CONTAMINATION = 'auto'
#####################################################################################################

############################### MANUSCRIPT VISUALIZATION ############################################
EXAMPLE_DIVE = 'example_dive'

PATH_TO_EXAMPLE_DIVE = UNSUPERVISED_LEARNING_DIR / f'{EXAMPLE_DIVE}'

EXAMPLE_DIRECTORY_WITH_PICKLED_ITEMS = PATH_TO_EXAMPLE_DIVE / 'pickled_items'

MANUSCRIPT_FIGURES_DIRECTORY = PATH_TO_EXAMPLE_DIVE / 'manuscript_figures'

MANUSCRIPT_FIG_SIZE = (12,8)
#####################################################################################################
