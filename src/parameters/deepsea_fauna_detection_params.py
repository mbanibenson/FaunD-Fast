from pathlib import Path

#################################### DATASET DIRECTORY ###################################
DIRECTORY_CONTAINING_IMAGE_DATASET = Path('/home/mbani/mardata/datasets/Pacific_dataset_for_fauna_detection/') #Must be hard coded. Organize images into dives/folders

########################## ##################################### #########################

########################## Segmentation, VAE training and outlier detection #########################
PROJECT_DIRECTORY = Path.cwd().parent

UNSUPERVISED_LEARNING_DIR = PROJECT_DIRECTORY / 'data/unsupervised_outlier_detection/' #Relative to src directory

DIVE_TO_DETECT_ANOMALIES = 'example_dive'

DIVE_PARENT_IMAGES_DIR = DIRECTORY_CONTAINING_IMAGE_DATASET / f'{DIVE_TO_DETECT_ANOMALIES}'

DIVE_WORKING_DIR = UNSUPERVISED_LEARNING_DIR / f'{DIVE_TO_DETECT_ANOMALIES}'
  
DIVE_SAMPLED_BACKGROUND_IMAGES_DIR = DIVE_WORKING_DIR / 'background_images'

DIVE_PICKLED_ITEMS_DIR = DIVE_WORKING_DIR / 'pickled_items'

DIVE_OUTPUT_DIR = DIVE_WORKING_DIR / 'detection_outputs'

NUMBER_OF_IMAGES_TO_SAMPLE_AS_BACKGROUND = 500

BATCH_SIZE = 32

LATENT_DIMENSION = 100

TRAINING_EPOCHS = 20

CONTAMINATION = 0.01
#####################################################################################################

########################## Supervised binary classification and annotation #########################
SUPERVISED_LEARNING_DIR = PROJECT_DIRECTORY / 'data/supervised_fauna_non_fauna_classification/' #Relative to src directory

DIVE_OUTPUT_DIR_AFTER_BINARY_CLASSIFICATION = SUPERVISED_LEARNING_DIR / 'classification_outputs/'

ANNOTATION_TOOL_DIR = PROJECT_DIRECTORY / 'custom_annotation_tool/'
#####################################################################################################

########################## TF OBJECT DETECTION API #########################
OBJECT_DETECTION_DIR = PROJECT_DIRECTORY / 'fauna_detection_with_tensorflow_object_detection_api'

############################### MANUSCRIPT VISUALIZATION ############################################
EXAMPLE_DIVE = 'example_dive'

PATH_TO_EXAMPLE_DIVE = UNSUPERVISED_LEARNING_DIR / f'{EXAMPLE_DIVE}'

EXAMPLE_DIRECTORY_WITH_PICKLED_ITEMS = PATH_TO_EXAMPLE_DIVE / 'pickled_items'

MANUSCRIPT_FIGURES_DIRECTORY = PATH_TO_EXAMPLE_DIVE / 'manuscript_figures'

MANUSCRIPT_FIG_SIZE = (12,8)

DETECTION_PATCHES_DIRECTORY = PATH_TO_EXAMPLE_DIVE / 'detection_patches'
#####################################################################################################
