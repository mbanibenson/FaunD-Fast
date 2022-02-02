import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd
import seaborn as sns

def visualize_embedded_segment_patches(embedded_feature_vectors, labels, combined_patches=None, figsize=(12,8)):
    '''
    Plot the embedding in 2D feature space
    
    '''
    fig, ax = plt.subplots(figsize=figsize)
    
    data_matrix = pd.DataFrame({'X':embedded_feature_vectors[:,0], 'Y':embedded_feature_vectors[:,1], 'Label':labels})
    
    #data_matrix['Label Names'] = data_matrix.Label.map({0:'Segment Patch', 1:'Support Set Patch', 2:'Test'})
    
    support_set_mappings =  {k, f'class_{k}' for k in sorted(labels)} #Merge background and support sets
    
    class_mappings = {**{0:'Background Patch'}, **support_set_mappings}
    
    data_matrix['Label Names'] = data_matrix.Label.map(class_mappings)
    
    #data_matrix = data_matrix.iloc[-4:]
    
    temp = sns.scatterplot(x='X', y='Y', hue='Label Names', data=data_matrix, ax=ax)
    
    if combined_patches:
        
        for x0, y0, patch in zip(data_matrix.X.values, data_matrix.Y.values, combined_patches):

            ab = AnnotationBbox(OffsetImage(patch, zoom=0.5), (x0, y0), frameon=False)

            ab.set_zorder(0)

            ax.add_artist(ab)
    
    return 