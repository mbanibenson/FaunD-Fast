from models.VAE_based_outlier_detection import train_Isolation_Forest_to_detect_anomalous_patches
from parameters import deepsea_fauna_detection_params

import pickle

from parameters import deepsea_fauna_detection_params

if __name__ == '__main__':
    
    directory_containing_pickled_items = deepsea_fauna_detection_params.DIVE_PICKLED_ITEMS_DIR
    
    contamination = deepsea_fauna_detection_params.CONTAMINATION
    
    with open(directory_containing_pickled_items / f'background_feature_vectors.pickle', 'rb') as f:
        
        background_feature_vectors = pickle.load(f)
        
        
    isolation_forest_outlier_detector = train_Isolation_Forest_to_detect_anomalous_patches(background_feature_vectors, contamination)
    
    print('Pickling outlier detector ...')
    with open(directory_containing_pickled_items / f'isolation_forest_outlier_detector.pickle', 'wb') as f:
        
        pickle.dump(isolation_forest_outlier_detector, f, pickle.HIGHEST_PROTOCOL)
        
    print('Done pickling outlier detector ...')