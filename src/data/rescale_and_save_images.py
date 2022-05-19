import pandas as pd
import numpy as np
from scipy.ndimage import zoom
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from skimage.io import imread, imsave
import shutil
from functools import partial
from skimage.util import crop as skimage_crop
from skimage.transform import resize
from pathlib import Path

    
def zoom_and_save(file_path, scaling_factor, zoomed_images_directory):
    '''
    Scale and crop images
    
    '''
    
    print(f'Scaling {file_path.name}')
    
    image = imread(file_path)
    
    image_name = file_path.name
    
    image_zoomed = zoom(image, [scaling_factor, scaling_factor, 1])
    
    imsave(zoomed_images_directory/image_name, image_zoomed.astype(np.uint8))
    
    return


def rescale_images( source_directory, scaling_factor=0.25):
    '''
    Rescale images to the same scale
    
    '''
    
    assert source_directory.is_dir(), 'Please specify source images folder as a directory'

    zoomed_images_directory = Path('/home/mbani/mardata/datasets/Pacific_dataset_for_fauna_detection')  / source_directory.name
    
    shutil.rmtree(zoomed_images_directory, ignore_errors=True)
    zoomed_images_directory.mkdir(parents=True, exist_ok=True)

    scale_and_save = partial(zoom_and_save, scaling_factor=scaling_factor, zoomed_images_directory=zoomed_images_directory)
    
    print (f'Scaling directory {source_directory.name} ...')
    
    with ProcessPoolExecutor(14) as executor:
        
        [executor.submit(scale_and_save, file_path) for file_path in source_directory.iterdir()]
    
    print('Finished ...')
    
    return

if __name__ == '__main__':
    
    directory_containing_subdirectories_with_test_images = Path('/home/mbani/mardata/datasets/Pacific_dataset')
    
    for directory_containing_test_images in directory_containing_subdirectories_with_test_images.iterdir():
    
        assert directory_containing_test_images.is_dir(), 'Please organize images into subdirectories'

        subdirectory_name = directory_containing_test_images.name

        exclude_list = ['dive_153', 'dive_117']

        if subdirectory_name not in exclude_list:

            continue
            
        print(f'Scaling images in {subdirectory_name} directory ...')

        #source_directory = Path('/home/mbani/mardata/datasets/Pacific_dataset/SO268-2_100-1_OFOS-05/')

        rescale_images(directory_containing_test_images, scaling_factor=0.25)
        
    print('Finished.')
    