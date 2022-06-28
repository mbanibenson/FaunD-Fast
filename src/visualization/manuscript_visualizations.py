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
from numpy.random import default_rng

rng = default_rng

plt.rcParams.update({'figure.max_open_warning': 0})

def generate_background_images_with_superpixel_overlays(path_to_pickled_segmented_images, path_to_pickled_original_images, directory_to_save_manuscript_plots, figsize, n_sample=10):
    '''
    Sample a few background images and segment them to indicate superpixel boundaries
    
    '''
    
    directory_to_save_manuscript_plots = directory_to_save_manuscript_plots / 'superpixel_boundary_overlays'
    
    directory_to_save_manuscript_plots.mkdir()
    
    
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


def generate_grid_view_of_background_superpixels(path_to_pickled_background_patches, directory_to_save_manuscript_plots, grid_dimension, figsize):
    '''
    Show a grod view of background superpixels
    
    '''
    with open(path_to_pickled_background_patches, 'rb') as f:
        
        list_of_background_patches = pickle.load(f)
    
    number_of_patches_to_show = grid_dimension * grid_dimension 
    
    list_of_background_patches = [patch for patch in list_of_background_patches]
    
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

        ab = AnnotationBbox(OffsetImage(patch, zoom=0.1), (x0, y0), frameon=False)

        ab.set_zorder(5)

        ax.add_artist(ab)
            
    file_name = directory_to_save_manuscript_plots / 'feature_space_view_of_background_superpixels.png'
            
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
    return


def generate_feature_space_view_of_all_flagged_anomalous_superpixels(path_to_detections_summary_table, directory_with_anomalous_superpixel_patches, path_to_pickled_pca_object, directory_to_save_manuscript_plots, figsize, thumbnails_only):
    '''
    Show projection of anomalous superpixels in feature space
    
    '''
    z_order = 5 if thumbnails_only else 0
    
    name = 'feature_space_view_of_all_anomalous_superpixels_thumbnails_only.png' if thumbnails_only else 'feature_space_view_of_all_anomalous_superpixels.png'
    
    show_legend = False if thumbnails_only else True
    
    
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
    
    temp = sns.scatterplot(x='X', y='Y', size='anomaly_score', hue='anomaly_score', sizes=(5,10), data=data_matrix, ax=ax, legend=show_legend)
        
    for x0, y0, patch in zip(data_matrix.X.values, data_matrix.Y.values, anomalous_superpixel_patches):

        ab = AnnotationBbox(OffsetImage(patch, zoom=0.05), (x0, y0), frameon=False)

        ab.set_zorder(z_order)

        ax.add_artist(ab)
            
    file_name = directory_to_save_manuscript_plots / name
            
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
    return


def generate_feature_space_view_of_top_k_anomalous_superpixels(path_to_detections_summary_table, directory_with_anomalous_superpixel_patches, path_to_pickled_pca_object, directory_to_save_manuscript_plots, figsize, k):
    '''
    Show projection of very anomalous superpixels in feature space
    
    '''
    
    with open(path_to_pickled_pca_object, 'rb') as f:
        
        pca_object = pickle.load(f)
        
        
    anomalous_superpixels_df = pd.read_csv(path_to_detections_summary_table)
    
    
    indices_that_sort_anomalous_scores = anomalous_superpixels_df.anomaly_score.argsort()
    
    top_k_anomalous_indices = indices_that_sort_anomalous_scores[:k]
    
    
    anomalous_superpixels_df = anomalous_superpixels_df.loc[top_k_anomalous_indices]
            
    anomalous_superpixels_feature_vectors = anomalous_superpixels_df.iloc[:,5:-2]
    
    anomalous_patch_file_paths = [(directory_with_anomalous_superpixel_patches / f'{fn}.png') for fn in anomalous_superpixels_df.patch_name.tolist()]
    
    assert len(anomalous_superpixels_feature_vectors) == len(anomalous_patch_file_paths), 'Number of superpixels does feature vectors not match patches'
    
    
    with ThreadPoolExecutor() as executor:
        
        anomalous_superpixel_patches = list(executor.map(imread, anomalous_patch_file_paths))
        
        
    print(f'Number of anomalous superpixels: {len(anomalous_superpixels_df)}')
    
    print(f'Anomalous superpixels shape: {anomalous_superpixels_feature_vectors.shape}')    
    
    print(f'Number of anomalous superpixel patches: {len(anomalous_superpixel_patches)}')

    anomalous_superpixels_feature_vectors_2d = pca_object.transform(anomalous_superpixels_feature_vectors)
            
    fig, ax = plt.subplots(figsize=figsize)
    
    data_matrix = pd.DataFrame({'X':anomalous_superpixels_feature_vectors_2d[:,0], 'Y':anomalous_superpixels_feature_vectors_2d[:,1], 'anomaly_score':anomalous_superpixels_df.anomaly_score.multiply(-1)})
    
    temp = sns.scatterplot(x='X', y='Y', data=data_matrix, ax=ax, legend=False)
        
    for x0, y0, patch in zip(data_matrix.X.values, data_matrix.Y.values, anomalous_superpixel_patches):

        ab = AnnotationBbox(OffsetImage(patch, zoom=0.1), (x0, y0), frameon=False)

        ab.set_zorder(5)

        ax.add_artist(ab)
            
    file_name = directory_to_save_manuscript_plots / f'feature_space_view_of_top_{k}_anomalous_superpixels.png'
            
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
    return


def generate_feature_space_view_of_anomalous_superpixels_after_binary_classification(path_to_post_classification_detections_summary_table, directory_with_anomalous_superpixel_patches, path_to_pickled_pca_object, directory_to_save_manuscript_plots, figsize):
    '''
    Show projection of anomalous superpixels in feature space
    
    '''

    anomalous_superpixels_df = pd.read_csv(path_to_post_classification_detections_summary_table)
    
    anomalous_superpixels_feature_vectors = anomalous_superpixels_df.loc[:,['pca_0','pca_1']]
    
    anomalous_patch_file_paths = [(directory_with_anomalous_superpixel_patches / f'{fn}.png') for fn in anomalous_superpixels_df.patch_name.tolist()]
    
    
    with ThreadPoolExecutor() as executor:
        
        anomalous_superpixel_patches = list(executor.map(imread, anomalous_patch_file_paths))
        
        
    print(f'Number of anomalous superpixels: {len(anomalous_superpixels_df)}')
    
    print(f'Anomalous superpixels shape: {anomalous_superpixels_feature_vectors.shape}')    
    
    print(f'Number of anomalous superpixel patches: {len(anomalous_superpixel_patches)}')
            
    fig, ax = plt.subplots(figsize=figsize)
    
    anomalous_superpixels_feature_vectors_2d = anomalous_superpixels_feature_vectors.to_numpy()
    
    data_matrix = pd.DataFrame({'X':anomalous_superpixels_feature_vectors_2d[:,0], 'Y':anomalous_superpixels_feature_vectors_2d[:,1], 'anomaly_score':anomalous_superpixels_df.anomaly_score.multiply(-1)})
    
    temp = sns.scatterplot(x='X', y='Y', size='anomaly_score', data=data_matrix, ax=ax)
        
    for x0, y0, patch in zip(data_matrix.X.values, data_matrix.Y.values, anomalous_superpixel_patches):

        ab = AnnotationBbox(OffsetImage(patch, zoom=0.1), (x0, y0), frameon=False)

        ab.set_zorder(5)

        ax.add_artist(ab)
            
    file_name = directory_to_save_manuscript_plots / 'feature_space_view_of_post_binary_classification_anomalous_superpixels.png'
            
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
    return

def generate_grid_view_of_anomalous_superpixels_after_binary_classification(directory_with_anomalous_superpixel_patches, directory_to_save_manuscript_plots, grid_dimension, figsize):
    '''
    Show a grid view of anomalous superpixels
    
    ''' 
    anomalous_patches_file_paths = list(directory_with_anomalous_superpixel_patches.iterdir())
    
    number_of_patches_to_show = grid_dimension * grid_dimension  
    
    sampled_anomalous_patches_file_paths = [fp for fp in anomalous_patches_file_paths if fp.stem.startswith('SO268-2_126')]
    
    if len(sampled_anomalous_patches_file_paths) > number_of_patches_to_show:
    
        sampled_anomalous_patches_file_paths = random.sample(sampled_anomalous_patches_file_paths, k=number_of_patches_to_show)
    
    sampled_anomalous_patches = list(map(imread, sampled_anomalous_patches_file_paths))
    
    sampled_anomalous_patches_to_visualize = np.stack(sampled_anomalous_patches)
    
    print(f'Anomalous patches shape {sampled_anomalous_patches_to_visualize.shape}')

    montage_of_anomalous_patches = montage(sampled_anomalous_patches_to_visualize, grid_shape=(grid_dimension, grid_dimension), channel_axis=-1, padding_width=1, fill=[1,1,1])
    
    fig, ax = plt.subplots(figsize=figsize)
        
    ax.imshow(montage_of_anomalous_patches)

    ax.set_xticks([])

    ax.set_yticks([])

    file_name = directory_to_save_manuscript_plots / f'grid_view_of_anomalous_superpixels_after_binary_classification.png'

    plt.savefig(file_name, dpi=300)
    
    return


def generate_screenshot_of_superpixel_annotation_tool(path_to_screenshot, directory_to_save_manuscript_plots, figsize):
    '''
    Show a grid view of anomalous superpixels
    
    ''' 
       
    screenshot = imread(path_to_screenshot)
    
    fig, ax = plt.subplots(figsize=figsize)
        
    ax.imshow(screenshot)

    ax.set_xticks([])

    ax.set_yticks([])

    file_name = directory_to_save_manuscript_plots / f'screenshot_of_superpixel_annotator.png'

    plt.savefig(file_name, dpi=300)
    
    return

def generate_distribution_of_annotated_morphotypes(path_to_annotated_datasheet,directory_to_save_manuscript_plots, figsize):
    '''
    Generate distributuion over annotated morphotypes
    
    '''
    annotated_datasheet = pd.read_csv(path_to_annotated_datasheet).sort_values(by='object_name')

    fig, ax = plt.subplots(figsize=figsize)
    
    sns.histplot(data=annotated_datasheet, x='object_name', ax=ax)
    
    ax.set_xlabel('Annotated Morphotypes')
    
    ax.set_ylabel('Absolute Count')
    
    ax.tick_params(axis='x', labelrotation=30)

    file_name = directory_to_save_manuscript_plots / f'distribution_of_annotated_morphotypes.png'

    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
    return



def generate_distribution_of_detected_morphotypes(path_to_detection_summary_table, directory_to_save_manuscript_plots, figsize):
    '''
    Generate distributuion over annotated morphotypes
    
    '''
    detections_datasheet = pd.read_csv(path_to_detection_summary_table).sort_values(by='object_class_name')

    fig, ax = plt.subplots(figsize=figsize)
    
    sns.histplot(data=detections_datasheet, x='object_class_name', ax=ax)
    
    ax.set_xlabel('Morphotype')
    
    ax.set_ylabel('Absolute Count')
    
    ax.tick_params(axis='x', labelrotation=30)

    file_name = directory_to_save_manuscript_plots / f'distribution_of_detected_morphotypes.png'

    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
    return


def generate_example_images_with_bounding_box_overlays(directory_with_example_detection_images, directory_to_save_manuscript_plots, figsize):
    '''
    Visualize a few detections
    
    ''' 
    directory_to_save_manuscript_plots = directory_to_save_manuscript_plots / 'bounding_box_detections'
    
    directory_to_save_manuscript_plots.mkdir()
    
    bbox_detection_images_file_paths = list(directory_with_example_detection_images.iterdir())
    
    bbox_detection_images = list(map(imread, bbox_detection_images_file_paths))
    
    for num, bbox_detection_image in enumerate(bbox_detection_images, start=1):
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.imshow(bbox_detection_image)
        
        ax.set_xticks([])
        
        ax.set_yticks([])
        
        file_name = directory_to_save_manuscript_plots / f'example_image_with_detection_bounding_box_overlay_{num}.png'
        
        plt.savefig(file_name, dpi=300)
        
    return
    
    
        
        