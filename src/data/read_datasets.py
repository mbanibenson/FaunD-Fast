from skimage.io import imread
from skimage.transform import rescale
from pathlib import Path
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import numpy as np


def read_individual_rgb_image(file_path):
    '''
    Load a single RGB image from disk, and optionally resize
    
    ARGUMENTS
    ---------
    file_path(str): Path-like location to the file on disk 
    
    scaling_factor(tuple): scaling factors to resize images. Provide a tuple for each channel
    
    '''
    img = imread(file_path)

    return img


def load_augmented_support_set_patches(directory_containing_support_sets, number_of_augmentations=200, target_size=(32,32)):
    '''
    Load augmented versions of support set patches
    
    '''
    subdirectories = [fp.name for fp in Path(directory_containing_support_sets).iterdir() if fp.is_dir()]
    
    # shutil.rmtree(directory_to_save_augmented_copies, ignore_errors=True)
    # Path(directory_to_save_augmented_copies).mkdir(exist_ok=True)
    
    generator_instance = ImageDataGenerator(
        rotation_range=45,
        horizontal_flip=True,
        vertical_flip=True,
        #brightness_range=(0.1,0.9),
        #width_shift_range=0.2,
        #height_shift_range=0.2,
    )
    
    #print(f'Augmenting {random.sample(subdirectories)} class')
    
    
    
    support_set_patches = []
    
    support_set_labels = []
    
    print('Generating Augmentations ...')
    
    for i, subdirectory in enumerate(subdirectories, start=1):
        
        print(f'Class being augmented {subdirectory}')
        
        support_set_data_generator = generator_instance.flow_from_directory(
        directory=directory_containing_support_sets,
        target_size=target_size,
        color_mode='rgb',
        class_mode='sparse',
        batch_size=number_of_augmentations,
        #save_to_dir=directory_to_save_augmented_copies,
        classes=[subdirectory]
        
        )
        
        number_of_items = 0
    
        for augmented_batch, _ in support_set_data_generator:

            #print(f'Augmenting class ...')
            
            support_set_patches.append(augmented_batch)
            
            #support_set_labels.extend([i]*len(augmented_batch))
            
            support_set_labels.extend([subdirectory]*len(augmented_batch))
            
            number_of_items += len(augmented_batch)
            
            if number_of_items >= 200:
                
                print(f'Total number of augmented {subdirectory} is {number_of_items}')

                break
                
    support_set_patches = np.concatenate(support_set_patches, axis=0)  
    
    support_set_patches = [arr.astype('uint8') for arr in support_set_patches]
    
    return support_set_patches, support_set_labels