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
import tensorflow as tf
from itertools import chain
from skimage.transform import resize, rescale
from torchvision.utils import draw_bounding_boxes
import torch
from scipy.ndimage import zoom

rng = default_rng

plt.rcParams.update({'figure.max_open_warning': 0})

def generate_background_images_with_superpixel_overlays(path_to_pickled_segmented_images, path_to_pickled_original_images, directory_to_save_manuscript_plots, figsize, n_sample=10):
    '''
    Sample a few background images and segment them to indicate superpixel boundaries
    
    '''
    
    figsize = (3, 2)
    
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
        
        fig.tight_layout()
        
        plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)
        
    return


def generate_grid_view_of_background_superpixels(path_to_pickled_background_patches, directory_to_save_manuscript_plots, grid_dimension, figsize):
    '''
    Show a grod view of background superpixels
    
    '''
    figsize = (4, 4)
    
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

    plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)
    
    return

def generate_feature_space_view_of_background_superpixels(path_to_pickled_background_feature_vectors, path_to_pickled_background_patches, path_to_pickled_pca_object, directory_to_save_manuscript_plots, figsize):
    '''
    Show projection of background superpixels in feature space
    
    '''
    figsize = (7, 5)
    
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
            
    plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)
    
    return


def generate_feature_space_view_of_all_flagged_anomalous_superpixels(path_to_detections_summary_table, directory_with_anomalous_superpixel_patches, path_to_pickled_pca_object, directory_to_save_manuscript_plots, figsize, thumbnails_only):
    '''
    Show projection of anomalous superpixels in feature space
    
    '''
    figsize = (7, 5)
    
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
            
    plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)
    
    return


def generate_feature_space_view_of_top_k_anomalous_superpixels(path_to_detections_summary_table, directory_with_anomalous_superpixel_patches, path_to_pickled_pca_object, directory_to_save_manuscript_plots, figsize, k):
    '''
    Show projection of very anomalous superpixels in feature space
    
    '''
    figsize = (7, 5)
    
    with open(path_to_pickled_pca_object, 'rb') as f:
        
        pca_object = pickle.load(f)
        
        
    anomalous_superpixels_df = pd.read_csv(path_to_detections_summary_table)
    
    anomaly_threshold = anomalous_superpixels_df.anomaly_score.quantile(0.25)
    
    selector = anomalous_superpixels_df.anomaly_score.le(anomaly_threshold)
    
    anomalous_superpixels_df = anomalous_superpixels_df.loc[selector]
    
    
#     indices_that_sort_anomalous_scores = anomalous_superpixels_df.anomaly_score.argsort()
    
#     top_k_anomalous_indices = indices_that_sort_anomalous_scores[:k]
    
    
#     anomalous_superpixels_df = anomalous_superpixels_df.loc[top_k_anomalous_indices]
    
            
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

        ab = AnnotationBbox(OffsetImage(patch, zoom=0.04), (x0, y0), frameon=False)

        ab.set_zorder(5)

        ax.add_artist(ab)
            
    file_name = directory_to_save_manuscript_plots / f'feature_space_view_of_top_{k}_anomalous_superpixels.png'
            
    plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)
    
    return


def generate_feature_space_view_of_anomalous_superpixels_after_binary_classification(path_to_post_classification_detections_summary_table, directory_with_anomalous_superpixel_patches, path_to_pickled_pca_object, directory_to_save_manuscript_plots, figsize):
    '''
    Show projection of anomalous superpixels in feature space
    
    '''
    figsize = (7, 5)

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
            
    plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)
    
    return

def generate_grid_view_of_anomalous_superpixels_after_binary_classification(directory_with_anomalous_superpixel_patches, directory_to_save_manuscript_plots, grid_dimension, figsize):
    '''
    Show a grid view of anomalous superpixels
    
    ''' 
    figsize = (4, 4)
    
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

    plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)
    
    return


def generate_grid_view_of_top_k_superpixels_based_on_anomaly_score(path_to_detections_summary_table, directory_with_anomalous_superpixel_patches, directory_to_save_manuscript_plots, grid_dimension, figsize):
    '''
    Show a grid view of anomalous superpixels
    
    ''' 
    figsize = (4, 4)
    
    k = grid_dimension * grid_dimension
        
    anomalous_superpixels_df = pd.read_csv(path_to_detections_summary_table)
    
    
    indices_that_sort_anomalous_scores = anomalous_superpixels_df.anomaly_score.argsort()
    
    top_k_anomalous_indices = indices_that_sort_anomalous_scores[:k]
    
    
    anomalous_superpixels_df = anomalous_superpixels_df.loc[top_k_anomalous_indices]
    
    anomalous_patch_file_paths = [(directory_with_anomalous_superpixel_patches / f'{fn}.png') for fn in anomalous_superpixels_df.patch_name.tolist()]
    
    
    with ThreadPoolExecutor() as executor:
        
        anomalous_superpixel_patches = list(executor.map(imread, anomalous_patch_file_paths))
        
        
    top_k_patches_to_visualize = np.stack(anomalous_superpixel_patches)
        
        
    montage_of_anomalous_patches = montage(top_k_patches_to_visualize, grid_shape=(grid_dimension, grid_dimension), channel_axis=-1, padding_width=1, fill=[1,1,1])
    
    fig, ax = plt.subplots(figsize=figsize)
        
    ax.imshow(montage_of_anomalous_patches)

    ax.set_xticks([])

    ax.set_yticks([])

    file_name = directory_to_save_manuscript_plots / f'grid_view_of__top_{k}_anomalous_superpixels.png'

    plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)
    
    return


def generate_screenshot_of_superpixel_annotation_tool(path_to_screenshot, directory_to_save_manuscript_plots, figsize):
    '''
    Show a grid view of anomalous superpixels
    
    ''' 
    figsize = (3.5, 2.8)
       
    screenshot = imread(path_to_screenshot)
    
    fig, ax = plt.subplots(figsize=figsize)
        
    ax.imshow(screenshot)

    ax.set_xticks([])

    ax.set_yticks([])

    file_name = directory_to_save_manuscript_plots / f'screenshot_of_superpixel_annotator.png'

    plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)
    
    return

def generate_distribution_of_annotated_morphotypes(path_to_annotated_datasheet,directory_to_save_manuscript_plots, figsize):
    '''
    Generate distributuion over annotated morphotypes
    
    '''
    figsize = (4, 3)
    
    annotated_datasheet = pd.read_csv(path_to_annotated_datasheet).sort_values(by='object_name')

    fig, ax = plt.subplots(figsize=figsize)
    
    sns.histplot(data=annotated_datasheet, x='object_name', ax=ax)
    
    ax.set_xlabel('Annotated Morphotypes')
    
    ax.set_ylabel('Absolute Count')
    
    #ax.tick_params(axis='x', labelrotation=30)
    
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right",
         rotation_mode="anchor")

    file_name = directory_to_save_manuscript_plots / f'distribution_of_annotated_morphotypes.png'

    plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)
    
    return



def generate_distribution_of_detected_morphotypes(path_to_detection_summary_table, directory_to_save_manuscript_plots, figsize):
    '''
    Generate distributuion over annotated morphotypes
    
    '''
    figsize = (4, 3)
    
    detections_datasheet = pd.read_csv(path_to_detection_summary_table).sort_values(by='object_class_name')

    fig, ax = plt.subplots(figsize=figsize)
    
    sns.histplot(data=detections_datasheet, x='object_class_name', ax=ax)
    
    ax.set_xlabel('Morphotype')
    
    ax.set_ylabel('Absolute Count')
    
   # ax.tick_params(axis='x', labelrotation=60)
    
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right",
         rotation_mode="anchor")
      

    file_name = directory_to_save_manuscript_plots / f'distribution_of_detected_morphotypes.png'

    plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)
    
    return


# def generate_example_images_with_bounding_box_overlays(directory_with_example_detection_images, directory_to_save_manuscript_plots, figsize):
#     '''
#     Visualize a few detections
    
#     ''' 
#     figsize = (2.5, 1.8)
    
#     directory_to_save_manuscript_plots = directory_to_save_manuscript_plots / 'bounding_box_detections'
    
#     directory_to_save_manuscript_plots.mkdir()
    
#     bbox_detection_images_file_paths = list(directory_with_example_detection_images.iterdir())
    
#     bbox_detection_images = list(map(imread, bbox_detection_images_file_paths))
    
#     for num, bbox_detection_image in enumerate(bbox_detection_images, start=1):
        
#         fig, ax = plt.subplots(figsize=figsize)
        
#         ax.imshow(bbox_detection_image)
        
#         ax.set_xticks([])
        
#         ax.set_yticks([])
        
#         file_name = directory_to_save_manuscript_plots / f'example_image_with_detection_bounding_box_overlay_{num}.png'
        
#         plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)
        
#     return


def generate_example_images_with_bounding_box_overlays(path_to_detection_summary_table, directory_to_save_manuscript_plots, figsize):
    '''
    Visualize a few detections
    
    ''' 
    figsize = (2.5, 1.8)
    
    directory_to_save_manuscript_plots = directory_to_save_manuscript_plots / 'bounding_box_detections'
    
    directory_to_save_manuscript_plots.mkdir(exist_ok=True)
    
    df = pd.read_csv(path_to_detection_summary_table)

    COLORS = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928'] * 2

    morphotype_classes = df.object_class_name.unique()

    color_mapping = {morphotype_class:color for morphotype_class, color in zip(morphotype_classes, COLORS)}

    df['bbox_color'] = df.object_class_name.map(color_mapping)


    for file_path, detections in df.groupby('parent_image_path'):
        
        file_path = Path(file_path)

        if (len(detections) > 3) and (len(detections) < 7):

            scaling_factor = 0.5

            image_array = imread(file_path)

            image_array = zoom(image_array, zoom=(scaling_factor, scaling_factor, 1)).astype('uint8')

            image_array = np.moveaxis(image_array, 2, 0)

            image_array = torch.from_numpy(image_array)

            boxes = (detections.loc[:,['xmin', 'ymin', 'xmax', 'ymax']].to_numpy() - [50,50,0,0]) * scaling_factor

            boxes = torch.from_numpy(boxes) 

            labels = detections.object_class_name.to_list()

            colors = detections.bbox_color.to_list()

            ttf = '/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf'

            image_array_with_bboxes = draw_bounding_boxes(image_array, boxes, labels,colors, font=ttf, font_size=17).numpy()

            image_array_with_bboxes = np.moveaxis(image_array_with_bboxes, 0, 2)

            fig, ax = plt.subplots(figsize = figsize)

            ax.imshow(image_array_with_bboxes)

            ax.set_xticks([])

            ax.set_yticks([])

            file_name = directory_to_save_manuscript_plots / f'{file_path.stem}_bbox_detections.png'

            plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)
            
            plt.close()
            
    return

def crop_detections_for_display(path_to_detections_summary_table):
    '''
    Crop the detections so they may be visualized
    
    '''
    detected_megafauna_df = pd.read_csv(path_to_detections_summary_table)
    
    morphotype_groupings = detected_megafauna_df.groupby('object_class_name')
    
    cropped_detections_to_display = {}
    
    for morphotype_class, morphotype_group in morphotype_groupings:
        
        morphotype_group_subset = morphotype_group.sort_values(by='score', ascending=False).iloc[:50]
        
        cropped_detections = []
        
        for detection in morphotype_group_subset.itertuples():
            
            image_array = imread(detection.parent_image_path)
            
            offset_height = int(detection.ymin)
            
            offset_width = int(detection.xmin)
            
            target_height = int(detection.ymax) - int(detection.ymin)
            
            target_width = int(detection.xmax) - int(detection.xmin)
            
            cropped_detection = tf.image.crop_to_bounding_box(image_array, offset_height, offset_width, target_height, target_width).numpy()
            
            cropped_detections.append(resize(cropped_detection, (96,96)))
            
        cropped_detections_to_display[morphotype_class] = cropped_detections
        
        print(f'{morphotype_class}: {len(cropped_detections)}')
        
    return cropped_detections_to_display


def generate_grid_view_of_megafauna_detected_by_faster_rcnn(path_to_detections_summary_table, directory_to_save_manuscript_plots, grid_dimension, figsize):
    '''
    Show a grid view of detected benthic mega fauna
    
    ''' 
    figsize = (4, 4)
    
    cropped_detections_to_display = crop_detections_for_display(path_to_detections_summary_table)
    
    print(f'crops for one group are {len(cropped_detections_to_display["Coral"])}')
    
    selected_cropped_detections_to_display = list(chain.from_iterable([cropped_detections_to_display[key][:grid_dimension] for key in cropped_detections_to_display.keys()]))
    
    print(f'selected crops are {len(selected_cropped_detections_to_display)}')
    
    cropped_detections_to_visualize = np.stack(selected_cropped_detections_to_display)
    
    number_of_morphotypes = len(cropped_detections_to_display.keys())
        
    montage_of_detected_morphotypes = montage(cropped_detections_to_visualize, grid_shape=(number_of_morphotypes, grid_dimension), channel_axis=-1, padding_width=1, fill=[1,1,1])
    
    morphotype_labels = '\n'.join(cropped_detections_to_display.keys())
    
    Path(directory_to_save_manuscript_plots / f'morphotype_labels_for_detection_grid.txt').write_text(morphotype_labels)
    
    fig, ax = plt.subplots(figsize=figsize)
        
    ax.imshow(montage_of_detected_morphotypes)

    ax.set_xticks([])

    ax.set_yticks([])

    file_name = directory_to_save_manuscript_plots / f'grid_view_of_detected_morphotypes.png'

    plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)
    
    return



def generate_grid_view_of_megafauna_from_pre_cropped_patches(path_to_cropped_patches, directory_to_save_manuscript_plots, figsize=(4,4)):
    '''
    Show a grid view of detected benthic mega fauna
    
    ''' 

    morphotype_labels = [morphotype_directory.name for morphotype_directory in path_to_cropped_patches.iterdir()]

    image_arrays_to_visualize = []

    grid_dimension = len(morphotype_labels)

    number_of_morphotypes = len(morphotype_labels)
    

    for morphotype_directory in sorted(path_to_cropped_patches.iterdir(), key=lambda x: (x.name)):

        file_paths = sorted(morphotype_directory.iterdir(), key=lambda x: float(x.stem.split('_')[0]), reverse=False)

        if morphotype_directory.name == 'Litter':

            file_paths = [file_paths[i] for i in [0,1,2,3,5,6,9,10,11,12,14,15,16]]

        selected_file_paths = file_paths[:grid_dimension]

        selected_image_arrays = [imread(fp) for fp in selected_file_paths]

        image_arrays_to_visualize.extend(selected_image_arrays)
        
        
    montage_of_detected_morphotypes = montage(image_arrays_to_visualize, grid_shape=(number_of_morphotypes, grid_dimension), channel_axis=-1, padding_width=1, fill=[1,1,1])

    Path(directory_to_save_manuscript_plots / f'morphotype_labels_for_detection_grid.txt').write_text('\n'.join(morphotype_labels))

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(montage_of_detected_morphotypes)

    ax.set_xticks([])

    ax.set_yticks([])

    file_name = directory_to_save_manuscript_plots / f'grid_view_of_detected_morphotypes.png'

    plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)
        
    
            

            
            
        
    
        
        