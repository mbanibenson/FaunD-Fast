from models.VAE_based_outlier_detection import detect_anomalous_patches_using_trained_Isolation_Forest
from parameters import deepsea_fauna_detection_params
from keras.models import load_model
import shutil

import pickle

from parameters import deepsea_fauna_detection_params

if __name__ == '__main__':
    
    directory_containing_pickled_items = deepsea_fauna_detection_params.DIVE_PICKLED_ITEMS_DIR
    
    directory_containing_test_images = deepsea_fauna_detection_params.DIVE_PARENT_IMAGES_DIR
    
    directory_to_save_patches_of_positive_detections = deepsea_fauna_detection_params.DIVE_OUTPUT_DIR
    
    shutil.rmtree(directory_to_save_patches_of_positive_detections, ignore_errors=True)
    directory_to_save_patches_of_positive_detections.mkdir(exist_ok=True)
    
    with open(directory_containing_pickled_items / f'isolation_forest_outlier_detector.pickle', 'rb') as f:
        
        trained_Isolation_Forest_model = pickle.load(f)
        
    VAE_model_file_path = str(directory_containing_pickled_items / 'trained_VAE_model')  
    
    trained_VAE_model = load_model(VAE_model_file_path)    
    
    detect_anomalous_patches_using_trained_Isolation_Forest(trained_VAE_model,
                                      trained_Isolation_Forest_model, 
                                      directory_containing_test_images,
                                      directory_to_save_patches_of_positive_detections)