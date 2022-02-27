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