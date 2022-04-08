from skimage.segmentation import slic, felzenszwalb
from skimage.measure import regionprops
import numpy as np
from numpy.random import default_rng
from skimage.transform import resize

rng = default_rng()

def generate_superpixels_using_slic(image_as_rgb, number_of_segments, compactness):
    '''
    Use slic algorithm to generate superpixels
    
    ARGUMENTS
    ---------
    image_as_rgb(ndarray): The image in memory as a numpy array 
    
    number_of_segments(int): Number of segments to generate 
    
    compactness(float): The compactness parameter
    
    RETURNS
    --------
    segmented_image(ndarray): A segmented image
    
    '''
    # segmented_image = slic(image_as_rgb, n_segments=number_of_segments, start_label=1, compactness=compactness, channel_axis=-1)
    
    segmented_image = felzenszwalb(image_as_rgb, scale=200, sigma=10, min_size=20)
    
    return segmented_image


def extract_image_patches_corresponding_to_the_superpixels(segmented_image, image_as_rgb, training_mode=True):
    '''
    Given a set of superpixels, extract the image patches within their bounding boxes
    
    ARGUMENTS
    ---------
    segmented_image(ndarray): The segmented image containing superpixels 
    
    image_as_rgb(ndarray): The image from which the segmentation was generated
    
    
    RETURNS
    -------
    segment_patches(ndarray): A batch of ndarrays corresponding to patches for each segment/superpixel
    
    '''
    region_properties = regionprops(label_image=segmented_image, intensity_image=image_as_rgb)

    slice_portions = [region_property.slice for region_property in region_properties]

    segment_patches = [image_as_rgb[slice_portion] for slice_portion in slice_portions]
    
    segment_patch_bounding_boxes = [region_property.bbox for region_property in region_properties]
    
    number_of_patches = len(segment_patches)
    
    indices_of_sampled_patches = rng.integers(0, number_of_patches, size=20)
    
    #sampled_segment_patches = np.asarray(segment_patches)[indices_of_sampled_patches].tolist()
    
    if training_mode:
    
        sampled_segment_patches = [segment_patches[i] for i in indices_of_sampled_patches]
    
        sampled_segment_patch_bounding_boxes = [segment_patch_bounding_boxes[i] for i in indices_of_sampled_patches]
        
    else:
        
        sampled_segment_patches = segment_patches
    
        sampled_segment_patch_bounding_boxes = segment_patch_bounding_boxes
    
    sampled_segment_patches =  [resize(patch, (32,32,3)) for patch in sampled_segment_patches]
    
    return sampled_segment_patches, sampled_segment_patch_bounding_boxes


