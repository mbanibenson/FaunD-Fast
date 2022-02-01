class underwater_image:
    '''
    Read an image and segment it
    
    '''
    def __init__(self, path_to_image):
        
        self.image_path = Path(path_to_image)
        
        self.rgb_image = None
        
        self.segmented_image = None
        
        self.segment_patches = None
        
        self.segment_patches_feature_vectors = None
        
        self.segment_patch_centroids = None
        
        self.georeferenced_coordinates_for_the_image = None
        
        
    def read_image(self):
        '''
        Load image from disk
        
        #data/read_datasets.py
        rescaled_image = read_individual_rgb_image(file_path, scaling_factors=None)
        
        '''
        file_path = self.image_path
        
        scaling_factors = (0.25,0.25,1)

        self.rgb_image = read_individual_rgb_image(file_path, scaling_factors=scaling_factors)
        
        return
    
    
    def segment_image(self):
        '''
        Perform segmentation on the image
        
        #features/segment_to_generate_superpixels.py
        segmented_image = generate_superpixels_using_slic(image_as_rgb, number_of_segments, compactness)
        
        '''
        image_as_rgb=self.rgb_image
        
        number_of_segments=250
        
        compactness=50

        self.segmented_image = generate_superpixels_using_slic(image_as_rgb, number_of_segments, compactness)
        
        return
    
    
    def extract_segmentation_patches_to_batch_of_ndarrays(self):
        '''
        Extract segment patches to numpy ndarray
        
        #features/segment_to_generate_superpixels.py
        segment_patches = extract_image_patches_corresponding_to_the_superpixels(segmented_image, image_as_rgb)
        
        '''
        segmented_image=self.segmented_image 
        
        image_as_rgb=self.rgb_image

        self.segment_patches = extract_image_patches_corresponding_to_the_superpixels(segmented_image, image_as_rgb)

        return
    
    
    def extract_features_from_segmentation_patches(self, feature_extractor_module_url=None, resize_dimension=None):
        '''
        Extract feature vectors from segmentation patches
        
        #features/segment_to_generate_superpixels.py
        extract_hand_engineered_hog_features_for_segmentation_patches(list_of_segment_patches)
        '''
        list_of_segment_patches = self.segment_patches

        self.segment_patches_feature_vectors = extract_hand_engineered_hog_features_for_segmentation_patches(list_of_segment_patches)

        return
    
    #################################### END OF CLASS METHODS #########################################
    
    
    

def segment_image_and_extract_segment_features(file_path, feature_extractor_module_url=None, resize_dimension=None):
    '''
    Create an instance of underwater, segment it and extract features from its superpixels
    
    '''
    
    #print(f'Processing image {file_path.name}', end='\r')
    
    print('Creating segmentation object ...', end='\r', flush=True)
    underwater_image_of_ccz = underwater_image(file_path) #ccz is the working area in the pacific
    
    print('Reading image to array ...', end='\r', flush=True)
    underwater_image_of_ccz.read_image()
    
    print('Segmenting the image ...', end='\r', flush=True)
    underwater_image_of_ccz.segment_image()
    
    print('Converting segment patches to ndarrays ...', end='\r', flush=True)
    underwater_image_of_ccz.extract_segmentation_patches_to_batch_of_ndarrays()
    
    print('Extract Features from the segments ...', end='\r', flush=True)
    underwater_image_of_ccz.extract_features_from_segmentation_patches(feature_extractor_module_url, resize_dimension)
    
    
    return underwater_image_of_ccz