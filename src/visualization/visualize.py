import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA



def visualize_embedded_segment_patches(embedded_feature_vectors, labels, combined_patches=None, figsize=(12,8), figname = None, directory_to_save_matplotlib_figures=None):
    '''
    Plot the embedding in 2D feature space
    
    '''
    if embedded_feature_vectors.shape[1] == 3:
        
        embedded_feature_vectors = PCA(2, whiten=True).fit_transform(embedded_feature_vectors)
        
    fig, ax = plt.subplots(figsize=figsize)
    
    data_matrix = pd.DataFrame({'X':embedded_feature_vectors[:,0], 'Y':embedded_feature_vectors[:,1], 'Label':labels})
    
    #data_matrix['Label Names'] = data_matrix.Label.map({0:'Segment Patch', 1:'Support Set Patch', 2:'Test'})
    
    # class_mappings =  {k:f'Support Set Patch' if k>0 else k:f'Background Patch' for k in sorted(labels)} #Merge background and support sets
    
    #class_mappings = {**{0:'Background Patch'}, **support_set_mappings}
    
    #data_matrix['Label Names'] = data_matrix.Label.map(lambda v: 'Background Patch' if int(v)==0 else 'Support Set Patch')
    
    data_matrix['Label Names'] = data_matrix.Label.map({0:'Background', 1:'Support Set', 2:'Detections'})
    
    #data_matrix = data_matrix.iloc[-4:]
    
    temp = sns.scatterplot(x='X', y='Y', hue='Label Names', data=data_matrix, ax=ax, s=5)
    
    if combined_patches:
        
        for x0, y0, patch in zip(data_matrix.X.values, data_matrix.Y.values, combined_patches):

            ab = AnnotationBbox(OffsetImage(patch, zoom=0.5), (x0, y0), frameon=False)

            ab.set_zorder(0)

            ax.add_artist(ab)
            
    plt.savefig(Path(directory_to_save_matplotlib_figures) / f'{figname}.png', dpi=150, bbox_inches='tight')
    
    return 

def visualize_absolute_count_of_detections_per_image(path_to_csv_with_detection_counts_per_image):
    '''
    Given the csv tabulating counts of detections per image, show shome visualizations
    
    '''
    
    path_to_csv_with_detection_counts_per_image = Path(path_to_csv_with_detection_counts_per_image)
    
    COLORS = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
    
    df_with_detection_counts_per_image = pd.read_csv(path_to_csv_with_detection_counts_per_image, parse_dates=['acquisition_time']).set_index('acquisition_time')
    
    column_colors = [] 
    
    fig, ax = plt.subplots(figsize=(20,6))
    
    
    
    for dive_number, sub_df in df_with_detection_counts_per_image.groupby('dive'):
    
        sub_df.plot.line(y='detection_counts', ax=ax, label=f'{dive_number}')
        
    ax.set_xlabel('Time of Image Acquisition')
    
    ax.set_ylabel('Number of Detections')
    
    plt.legend() 
        
    plt.savefig(path_to_csv_with_detection_counts_per_image.parents[0]/'counts_per_detection.png', dpi=300)
    
    return


def visualize_distribution_over_count_of_detections_per_dive(path_to_csv_with_detection_counts_per_image):
    '''
    Given the csv tabulating counts of detections per image, show shome visualizations
    
    '''
    
    path_to_csv_with_detection_counts_per_image = Path(path_to_csv_with_detection_counts_per_image)
    
    COLORS = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
    
    df_with_detection_counts_per_image = pd.read_csv(path_to_csv_with_detection_counts_per_image, parse_dates=['acquisition_time']).set_index('acquisition_time')
    
    column_colors = [] 
    
    #fig, ax = plt.subplots(figsize=(20,8))
    
    g = sns.displot(data=df_with_detection_counts_per_image, x='detection_counts', hue='dive')
            
    g.set_axes_labels('Number of Detections', 'Density')

    plt.savefig(path_to_csv_with_detection_counts_per_image.parents[0]/'distribution_over_counts_per_detection.png', dpi=300)
    
    return


if __name__ == '__main__':
    
    path_to_csv_with_detection_counts_per_image = '/home/mbani/mardata/datasets/positively_detected_fauna_experimental/table_with_detection_counts_per_image.csv'
    
    visualize_absolute_count_of_detections_per_image(path_to_csv_with_detection_counts_per_image)
    
    visualize_distribution_over_count_of_detections_per_dive(path_to_csv_with_detection_counts_per_image)
    
    
    
    