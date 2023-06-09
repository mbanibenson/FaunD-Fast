import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from skimage.io import imread
from math import ceil
import shutil
from skimage.transform import resize
import random
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox



################################## Data Loaders ################################
def load_data_into_training_and_validation_sets(directory_containing_datasets, target_size):
    '''
    Given a directory with subdirectories of images (each a different class), load them into train and validation sets
    
    '''
    directory_containing_datasets = Path(directory_containing_datasets)
    
    rescale = None
    
    data_datagen = ImageDataGenerator(rescale=rescale, rotation_range=60, horizontal_flip=True, vertical_flip=True, validation_split=0.2)

    train_generator = data_datagen.flow_from_directory(directory_containing_datasets,
                                                       target_size=(target_size,target_size),
                                                       batch_size=32,
                                                       class_mode='categorical',
                                                       classes=['fauna', 'non_fauna'],
                                                       subset='training'
                                                      )
    
    validation_generator = data_datagen.flow_from_directory(directory_containing_datasets,
                                                       target_size=(target_size,target_size),
                                                       batch_size=32,
                                                       class_mode='categorical',
                                                       classes=['fauna', 'non_fauna'],
                                                       subset='validation')
    
    class_label_mappings = {val:key for key,val in train_generator.class_indices.items()}
    
    return train_generator, validation_generator, class_label_mappings



def load_images_for_sorting(directory_containing_images_to_sort, target_size):
    '''
    Load images together with their file paths for predictions and sorting into fauna and non-fauna
    
    '''
    data_gen = ImageDataGenerator()
    
    batch_size = 64
    
    test_data_gen = data_gen.flow_from_directory(directory_containing_images_to_sort,
                                                target_size=(target_size,target_size),
                                                batch_size=batch_size,
                                                class_mode=None,
                                                shuffle=False,
                                                classes = ['patches']
                                                )
    
    test_data_file_paths = [Path(fp) for fp in test_data_gen.filepaths]
    
    number_of_batches = ceil(len(test_data_file_paths) / batch_size)
        
    return test_data_gen, test_data_file_paths, number_of_batches
################################## End of Data Loaders ############################################


################################## CNN Model specification ######################################
def build_model(number_of_classes, target_size):
    '''
    define and build the architecture of our model
    
    ## Adapted from https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
    
    '''
    inputs = layers.Input(shape=(target_size,target_size,3))
    
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights='imagenet')
    
    model.trainable = False
    
    x = layers.GlobalAveragePooling2D(name='avg_pool')(model.output)
    
#     x = layers.BatchNormalization()(x)
    
#     top_dropout_rate=0.2
    
#     x = layers.Dropout(top_dropout_rate, name='top_dropout')(x)
    
    outputs = layers.Dense(number_of_classes, activation='softmax', name='pred')(x)
    
    
    model = tf.keras.Model(inputs, outputs, name='EfficientNet')
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def train_model(model, train_generator, validation_generator):
    '''
    Train the given model given train and validation generators
    
    '''

    history = model.fit(train_generator,
                     steps_per_epoch=20,
                     epochs=50,
                     validation_data=validation_generator,
                     #validation_steps=20,
                     #class_weight={0:2, 1:1},
                     verbose=2)
    
    return model, history

################################## End of CNN Model specification ################################


################################## Visualization utils ##################################
def visualize_model_curves(history, figsize=(12,8), directory_to_save_matplotlib_figures=None):
    '''
    Given history object generated from training, plot the training curves
    
    '''
    training_accuracies = history.history['accuracy']
    
    validation_accuracies = history.history['val_accuracy']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(training_accuracies, label='training')
    
    ax.plot(validation_accuracies, label='validation')
    
    ax.set_xlabel('Epoch')
    
    ax.set_ylabel('Accuracy')
    
    ax.legend()
    
    plt.savefig(Path(directory_to_save_matplotlib_figures) / f'model_curves.png', dpi=150, bbox_inches='tight')
    
    
def visualize_distribution_over_predictions(list_of_predictions, figsize=(12,8), figname=None, directory_to_save_matplotlib_figures=None):
    '''
    Visualize proportions of test set classified under each class
    
    '''
    patch_predictions_df = pd.DataFrame({'predictions':list_of_predictions})

    patch_predictions_df_with_counts = patch_predictions_df.groupby('predictions').size().to_frame('items').reset_index()

    ax = patch_predictions_df_with_counts.plot(kind='bar', x='predictions', y='items', 
                                               rot=0, figsize=figsize, legend=False,
                                              xlabel='Classes', ylabel='Count')
    
    rects = ax.patches

    # Make some labels.
    labels = [i for i in patch_predictions_df_with_counts['items']]

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(
        rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom")
        
    plt.savefig(Path(directory_to_save_matplotlib_figures) / f'distributon_over_predictions_for_{figname}.png', dpi=150, bbox_inches='tight')
        
    return
        
        
def visualize_embedded_segment_patches(path_to_data_csv, directory_containing_patches, figsize=(12,8), figname = None, directory_to_save_matplotlib_figures=None):
    '''
    Plot the embedding in 2D feature space
    
    '''      
    fig, ax = plt.subplots(figsize=figsize)
    
    df = pd.read_csv(path_to_data_csv)
    
    temp = sns.scatterplot(x='pca_0', y='pca_1', data=df, ax=ax, s=5)
    
    directory_containing_patches = Path(directory_containing_patches)
    
    patch_file_paths = [directory_containing_patches/f'{patch_name}.png' for patch_name in df.patch_name]
    
    patches = [imread(fp) for fp in patch_file_paths]
            
    for x0, y0, patch in zip(df.pca_0.values, df.pca_1.values, patches):

        ab = AnnotationBbox(OffsetImage(patch, zoom=0.4), (x0, y0), frameon=False)

        ab.set_zorder(2)

        ax.add_artist(ab)
            
    plt.savefig(Path(directory_to_save_matplotlib_figures) / f'{figname}.png', dpi=150, bbox_inches='tight')
    
    return    
################################## End of Visualization of model curves ##################################


############################## Sorting utils #########################################
def sort_images_using_trained_model(trained_model, class_label_mappings, directory_containing_images_to_sort, directory_to_save_sorted_images, target_size):
    '''
    Make predictions on test images and sort them based on their predicted class labels
    
    '''
    directory_containing_images_to_sort = Path(directory_containing_images_to_sort)
    
    directory_to_save_sorted_images = Path(directory_to_save_sorted_images)
    
        
    test_data_gen, file_paths, number_of_batches = load_images_for_sorting(directory_containing_images_to_sort, target_size)
    
    file_names_for_sorting = []
    
    
    predicted_classes = trained_model.predict(test_data_gen, 
                                              steps=number_of_batches)
    
    predicted_classes = np.argmax(predicted_classes, axis=1)
    
    predicted_classes_with_string_labels = [class_label_mappings[i] for i in predicted_classes]
          
    for file_name, predicted_class in zip(file_paths, predicted_classes_with_string_labels):
        
        file_name_for_sorting = directory_to_save_sorted_images / f'{str(predicted_class)}/{file_name.name}'
        
        file_names_for_sorting.append(file_name_for_sorting)
      
    
    #Create the directories to save sorted images
    unique_classes = set(predicted_classes_with_string_labels)
    
        
    [(directory_to_save_sorted_images / f'{unique_class}').mkdir(exist_ok=True) for unique_class in unique_classes]
    
    with ThreadPoolExecutor() as executor:
        
        [executor.submit(shutil.copy2, source_fn, sorted_fn) for source_fn, sorted_fn in zip(file_paths, file_names_for_sorting)]
    
    #predicted_classes_with_string_labels = [class_label_mappings[i] for i in predicted_classes]
    
    print(f'Unique classes: \n {pd.Series(predicted_classes_with_string_labels).value_counts()}')
    
    return predicted_classes_with_string_labels



############################## End of sorting utils #########################################

############################## MAIN FUNCTION CALL ###########################################
def train_fauna_non_fauna_classifier_and_auto_sort_anomalous_patches(directory_containing_training_datasets, target_size, number_of_classes, figsize, directory_to_save_matplotlib_figures, directory_containing_unsupervised_outlier_detection_results, directory_to_save_sorted_images):
    '''
    Train a binary classifier and use it to automatically sort detected anomalous patches
    
    '''
    #Load datasets
    train_generator, validation_generator, class_label_mappings = load_data_into_training_and_validation_sets(directory_containing_training_datasets, target_size)

    #Set up the model
    model = build_model(number_of_classes, target_size)

    #Perform model training
    trained_model, history = train_model(model, train_generator, validation_generator)
    
    #Visualize training curves
    visualize_model_curves(history, figsize=figsize, directory_to_save_matplotlib_figures=directory_to_save_matplotlib_figures)
    
    
    for dive in directory_containing_unsupervised_outlier_detection_results.iterdir():
        
        dive_to_sort = dive.name
        
        directory_containing_images_to_sort = directory_containing_unsupervised_outlier_detection_results /f'{dive_to_sort}/detection_outputs'
        
        if directory_containing_images_to_sort.exists():
            
            print(f'Sorting {dive_to_sort} ...')

            #Perform sorting (clasification inference)
            list_of_predictions = sort_images_using_trained_model(trained_model, class_label_mappings, directory_containing_images_to_sort, directory_to_save_sorted_images, target_size)

            #Visualize distribuion over predictions
            visualize_distribution_over_predictions(list_of_predictions, figsize=(12,8), figname=dive_to_sort, directory_to_save_matplotlib_figures=directory_to_save_matplotlib_figures)

            #visualize_embedded_segment_patches(path_to_data_csv, directory_containing_patches, figsize=(20,12), figname = figname, directory_to_save_matplotlib_figures=directory_to_save_matplotlib_figures)
            
#############################################################################################


# if __name__ == '__main__':
    
#     #Set variables
#     target_size = 224
#     number_of_classes = 2
#     figsize = (12,8)
#     dive_to_sort = 'random_dive'
    
#     data_directory = Path.cwd().parents[0] / 'data'
    
#     directory_containing_training_datasets = data_directory / 'supervised_fauna_non_fauna_classification/training_dataset/'
    
#     directory_containing_classification_outputs = data_directory / 'supervised_fauna_non_fauna_classification/classification_outputs'
    
#     shutil.rmtree(directory_containing_classification_outputs, ignore_errors=True)
#     directory_containing_classification_outputs.mkdir()
    
#     directory_to_save_sorted_images = directory_containing_classification_outputs / 'classified_patches'
#     directory_to_save_sorted_images.mkdir()

#     path_to_post_processed_summary_table = directory_containing_classification_outputs / 'master_detections_summary_table.csv'
    
#     directory_to_save_matplotlib_figures = directory_containing_classification_outputs / 'matplotlib_figures'
#     directory_to_save_matplotlib_figures.mkdir()
    
#     directory_containing_pure_fauna_patches = directory_to_save_sorted_images / 'fauna'
    
#     directory_containing_unsupervised_outlier_detection_results = data_directory / 'unsupervised_outlier_detection'
    
#     #Load datasets
#     train_generator, validation_generator, class_label_mappings = load_data_into_training_and_validation_sets(directory_containing_training_datasets, target_size)

#     #Set up the model
#     model = build_model(number_of_classes, target_size)

#     #Perform model training
#     trained_model, history = train_model(model, train_generator, validation_generator)
    
#     #Visualize training curves
#     visualize_model_curves(history, figsize=figsize, directory_to_save_matplotlib_figures=directory_to_save_matplotlib_figures)
    
    
#     for dive in directory_containing_unsupervised_outlier_detection_results.iterdir():
        
#         dive_to_sort = dive.name
        
#         directory_containing_images_to_sort = directory_containing_unsupervised_outlier_detection_results /f'{dive_to_sort}/detection_outputs'
        
#         if directory_containing_images_to_sort.exists():
            
#             print(f'Sorting {dive_to_sort} ...')

#             #Perform sorting (clasification inference)
#             list_of_predictions = sort_images_using_trained_model(trained_model, class_label_mappings, directory_containing_images_to_sort, directory_to_save_sorted_images, target_size)

#             #Visualize distribuion over predictions
#             visualize_distribution_over_predictions(list_of_predictions, figsize=(12,8), figname=dive_to_sort, directory_to_save_matplotlib_figures=directory_to_save_matplotlib_figures)

#             #visualize_embedded_segment_patches(path_to_data_csv, directory_containing_patches, figsize=(20,12), figname = figname, directory_to_save_matplotlib_figures=directory_to_save_matplotlib_figures)
            
    
    
    