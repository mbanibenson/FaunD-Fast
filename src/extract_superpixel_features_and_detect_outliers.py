from models.VAE_based_outlier_detection import train_model
from models.VAE_based_outlier_detection import detect_outliers_using_trained_VAE
import shutil
import time
from parameters import deepsea_fauna_detection_params
import pickle

if __name__ == '__main__':
     
    working_directory = deepsea_fauna_detection_params.DIVE_WORKING_DIR
    
    working_directory.mkdir(exist_ok = True)

    directory_containing_sampled_training_images = deepsea_fauna_detection_params.DIVE_SAMPLED_BACKGROUND_IMAGES_DIR

    directory_containing_parent_images = deepsea_fauna_detection_params.DIVE_PARENT_IMAGES_DIR

    outputs_directory = deepsea_fauna_detection_params.DIVE_OUTPUT_DIR 
    
    directory_containing_pickled_items = deepsea_fauna_detection_params.DIVE_PICKLED_ITEMS_DIR
    
    directory_containing_pickled_items.mkdir(exist_ok=True)
    
    latent_dimension = deepsea_fauna_detection_params.LATENT_DIMENSION

    epochs = deepsea_fauna_detection_params.TRAINING_EPOCHS
    
    batch_size = deepsea_fauna_detection_params.BATCH_SIZE

    contamination = deepsea_fauna_detection_params.CONTAMINATION

    target_image_height, target_image_width = 96, 96

    
    
    
    print('Copying training images ...')
    copy_training_images_from_parent_images(directory_containing_parent_images, directory_containing_sampled_training_images, sample_size=400)
    print('Finished copying training images ...')

#     shutil.rmtree(outputs_directory, ignore_errors=True)
#     outputs_directory.mkdir(exist_ok=True)

#     directory_to_save_patches_of_positive_detections = outputs_directory

#     directory_to_save_matplotlib_figures = outputs_directory

#     #shutil.rmtree(directory_to_save_matplotlib_figures, ignore_errors=True)

#     #directory_to_save_matplotlib_figures.mkdir(exist_ok=True)

#     training_tic = time.time()

#     trained_VAE_model, train_generator, number_of_train_batches, segmented_images, original_images, background_feature_vectors, background_patches, pca_object = train_model(directory_containing_sampled_training_images, batch_size, epochs, latent_dimension, directory_to_save_matplotlib_figures)

#     training_toc = time.time()


#     inference_tic = time.time()



#     print('Detecting outliers using trained VAE ...')
#     #Detect outliers in test images
#     detect_outliers_using_trained_VAE(trained_VAE_model, train_generator, 
#                                       number_of_train_batches,directory_containing_parent_images,
#                                       directory_to_save_patches_of_positive_detections,
#                                       batch_size=batch_size, im_height=target_image_height, 
#                                       im_width=target_image_width, pca_object=None, 
#                                       contamination=contamination)
#     inference_toc = time.time()
    
    
#     print('Pickling things ...\n')
    
#     print('Pickling original images ...')
#     with open(directory_containing_pickled_items / f'original_images.pickle', 'wb') as f:
        
#         pickle.dump(original_images, f, pickle.HIGHEST_PROTOCOL)
        
#     print('Pickling segmented images ...')
#     with open(directory_containing_pickled_items / f'segmented_images.pickle', 'wb') as f:
        
#         pickle.dump(segmented_images, f, pickle.HIGHEST_PROTOCOL)
    
#     print('Pickling background feature vectors ...')
#     with open(directory_containing_pickled_items / f'background_feature_vectors.pickle', 'wb') as f:
        
#         pickle.dump(background_feature_vectors, f, pickle.HIGHEST_PROTOCOL)
        
#     print('Pickling background patches ...')
#     with open(directory_containing_pickled_items / f'background_patches.pickle', 'wb') as f:
        
#         pickle.dump(background_patches, f, pickle.HIGHEST_PROTOCOL)
        
#     print('Pickling pca ...')
#     with open(directory_containing_pickled_items / f'pca_object.pickle', 'wb') as f:
        
#         pickle.dump(pca_object, f, pickle.HIGHEST_PROTOCOL)
        
#     print('\nDone pickling things ...')


#     visualization_tic = time.time()

# #     print('Extracting features for training sets ...')
# #     #Extract features from training set for later visualization
# #     training_features, pca_object = extract_features_using_trained_VAE(trained_VAE_model, train_generator,number_of_train_batches,
# #                                                                          batch_size=batch_size, im_height=target_image_height, im_width=target_image_width)

# #     print('Generating visualization for outliers ...')
# #     #Visualize outliers
# #     visualize_embedded_segment_patches(outlier_features, outlier_patches, figsize=(20,12),points_only=False, figname = 'outliers', directory_to_save_matplotlib_figures=directory_to_save_matplotlib_figures)

# #     print('Generating visualization for training sets without patches ...')
# #     #Visualize the training set without patches
# #     visualize_embedded_segment_patches(training_features, list_of_training_patches, figsize=(20,12),points_only=True, figname = 'training_set_scatter_plot', directory_to_save_matplotlib_figures=directory_to_save_matplotlib_figures)

# #     print('Generating visualization for training sets with patches ...')
# #     #Visualize the training set with patches
# #     visualize_embedded_segment_patches(training_features, list_of_training_patches, figsize=(20,12),points_only=False, figname = 'training_set_with_patches', directory_to_save_matplotlib_figures=directory_to_save_matplotlib_figures)

#     visualization_toc = time.time()

#     visualization_time_taken = time.gmtime(visualization_toc-visualization_tic)

#     training_time_taken = time.gmtime(training_toc-training_tic)

#     inference_time_taken = time.gmtime(inference_toc-inference_tic)



#     with open(directory_to_save_matplotlib_figures / 'processing_time.txt', 'w') as file:

#         print(f'Completed Training in {training_time_taken.tm_hour} Hours, {training_time_taken.tm_min} Minutes and {training_time_taken.tm_sec} Seconds \n', file=file)

#         print(f'Completed Inference in {inference_time_taken.tm_hour} Hours, {inference_time_taken.tm_min} Minutes and {inference_time_taken.tm_sec} Seconds \n', file=file)

#         print(f'Completed Visualizations in {visualization_time_taken.tm_hour} Hours, {visualization_time_taken.tm_min} Minutes and {visualization_time_taken.tm_sec} Seconds \n', file=file)