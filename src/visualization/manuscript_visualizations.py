import matplotlib
matplotlib.use('agg')

from pathlib import Path
import pickle
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import random
from skimage.util import montage
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import seaborn as sns
from skimage.io import imread
import random

def generate_background_images_with_superpixel_overlays(path_to_pickled_segmented_images, path_to_pickled_original_images, directory_to_save_manuscript_plots, figsize, n_sample=10):
    '''
    Sample a few background images and segment them to indicate superpixel boundaries
    
    '''
    with open(path_to_pickled_segmented_images, 'rb') as f:
        
        list_of_segmented_images = pickle.load(f)
        
        
    with open(path_to_pickled_original_images, 'rb') as f:
        
        list_of_original_images = pickle.load(f)
        
    
    pairs_of_original_and_segmented_images = list(zip(list_of_original_images, list_of_segmented_images))
    
    for num, (original_image, segmented_image) in enumerate(pairs_of_original_and_segmented_images, start=1):
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.imshow(mark_boundaries(original_image, segmented_image))
        
        ax.set_xticks([])
        
        ax.set_yticks([])
        
        file_name = directory_to_save_manuscript_plots / f'background_images_with_superpixel_boundaries_{num}.png'
        
        plt.savefig(file_name, dpi=300)
        
    return


def generate_grid_view_of_background_superpixels(path_to_pickled_background_patches, directory_to_save_manuscript_plots, figsize):
    '''
    Show a grod view of background superpixels
    
    '''
    with open(path_to_pickled_background_patches, 'rb') as f:
        
        list_of_background_patches = pickle.load(f)
    
    grid_dimension = 10
    
    number_of_patches_to_show = grid_dimension * grid_dimension    
    
    sampled_background_patches = random.sample(list_of_background_patches, k=number_of_patches_to_show)
    
    background_patches_to_visualize = np.stack(sampled_background_patches)
    
    print(f'Background patches shape {background_patches_to_visualize.shape}')

    montage_of_background_patches = montage(background_patches_to_visualize, grid_shape=(grid_dimension, grid_dimension), channel_axis=-1, padding_width=1, fill=[1,1,1])
    
    fig, ax = plt.subplots(figsize=figsize)
        
    ax.imshow(montage_of_background_patches)

    ax.set_xticks([])

    ax.set_yticks([])

    file_name = directory_to_save_manuscript_plots / f'grid_view_of_background_superpixels.png'

    plt.savefig(file_name, dpi=300)
    
    return

def generate_feature_space_view_of_background_superpixels(path_to_pickled_background_feature_vectors, path_to_pickled_background_patches, path_to_pickled_pca_object, directory_to_save_manuscript_plots, figsize):
    '''
    Show projection of background superpixels in feature space
    
    '''
    
    with open(path_to_pickled_background_feature_vectors, 'rb') as f:
        
        background_feature_vectors = pickle.load(f)
        
        
    with open(path_to_pickled_background_patches, 'rb') as f:
        
        list_of_background_patches = pickle.load(f)
        
        
    with open(path_to_pickled_pca_object, 'rb') as f:
        
        pca_object = pickle.load(f)
        
        
    print(f'Background feature vectors shape: {background_feature_vectors.shape}')
    
    print(f'Number of background patches: {len(list_of_background_patches)}')
    
    background_feature_vectors_2d = pca_object.transform(background_feature_vectors)
            
    fig, ax = plt.subplots(figsize=figsize)
    
    data_matrix = pd.DataFrame({'X':background_feature_vectors_2d[:,0], 'Y':background_feature_vectors_2d[:,1]})
    
    temp = sns.scatterplot(x='X', y='Y', data=data_matrix, ax=ax, s=5)
        
    for x0, y0, patch in zip(data_matrix.X.values, data_matrix.Y.values, list_of_background_patches):

        ab = AnnotationBbox(OffsetImage(patch, zoom=0.3), (x0, y0), frameon=False)

        ab.set_zorder(0)

        ax.add_artist(ab)
            
    file_name = directory_to_save_manuscript_plots / 'feature_space_view_of_background_superpixels.png'
            
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
    return


def generate_feature_space_view_of_all_flagged_anomalous_superpixels(path_to_detections_summary_table, directory_with_anomalous_superpixel_patches, path_to_pickled_pca_object, directory_to_save_manuscript_plots, figsize):
    '''
    Show projection of background superpixels in feature space
    
    '''
    
    with open(path_to_pickled_pca_object, 'rb') as f:
        
        pca_object = pickle.load(f)
        
        
    anomalous_superpixels_df = pd.read_csv(path_to_detections_summary_table)
    
    anomalous_superpixels_feature_vectors = anomalous_superpixels_df.iloc[:,5:-2]
    
    anomalous_patch_file_paths = [(directory_with_anomalous_superpixel_patches / f'{fn}.png') for fn in anomalous_superpixels_df.patch_name.tolist()]
    
    
    with ThreadPoolExecutor() as executor:
        
        anomalous_superpixel_patches = list(executor.map(imread, anomalous_patch_file_paths))
        
        
    print(f'Number of anomalous superpixels: {len(anomalous_superpixels_df)}')
    
    print(f'Anomalous superpixels shape: {anomalous_superpixels_feature_vectors.shape}')    
    
    print(f'Number of anomalous superpixel patches: {len(anomalous_superpixel_patches)}')

    anomalous_superpixels_feature_vectors_2d = pca_object.transform(anomalous_superpixels_feature_vectors)
            
    fig, ax = plt.subplots(figsize=figsize)
    
    data_matrix = pd.DataFrame({'X':anomalous_superpixels_feature_vectors_2d[:,0], 'Y':anomalous_superpixels_feature_vectors_2d[:,1], 'anomaly_score':anomalous_superpixels_df.anomaly_score.multiply(-1)})
    
    temp = sns.scatterplot(x='X', y='Y', size='anomaly_score', data=data_matrix, ax=ax)
        
    for x0, y0, patch in zip(data_matrix.X.values, data_matrix.Y.values, anomalous_superpixel_patches):

        ab = AnnotationBbox(OffsetImage(patch, zoom=0.1), (x0, y0), frameon=False)

        ab.set_zorder(0)

        ax.add_artist(ab)
            
    file_name = directory_to_save_manuscript_plots / 'feature_space_view_of_all_anomalous_superpixels.png'
            
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
    return
    
    
        
        