from skimage.segmentation import slic
from skimage.measure import regionprops

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
    segmented_image = slic(image_as_rgb, n_segments=number_of_segments, start_label=1, compactness=compactness)
    
    return segmented_image


def extract_image_patches_corresponding_to_the_superpixels(segmented_image, image_as_rgb):
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
        
    return segment_patches


