import numpy as np
from itertools import chain
from skimage.transform import resize, rescale
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import norm
from scipy.optimize import nnls
from scipy.optimize import minimize
from numpy.random import default_rng
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score
from sklearn.decomposition import PCA

rng = default_rng()

import sys
sys.path.append('./')
from models.core_utils import merge_segmentation_patches_from_all_images
    
    

def embedd_segment_feature_vectors_using_supervised_pca(segmented_image_objects, support_set_feature_vectors, support_set_patches, support_set_labels):
    '''
    Embedd feature vectors to 2D manifold
    
    '''
    segmentation_feature_vectors, segment_patches, _ , _ = merge_segmentation_patches_from_all_images(segmented_image_objects)
    
#     random_sample_indices = rng.integers(0, len(segmented_image_objects), size=100)
    
#     segmentation_feature_vectors = segmentation_feature_vectors[random_sample_indices]
    
#     segment_patches = np.array(segment_patches)[random_sample_indices].tolist()
    
    segmentation_feature_vectors_labels = np.zeros(shape=(len(segmentation_feature_vectors),))
    
    support_set_feature_vectors_labels = support_set_labels
    
    combined_patches = segment_patches + support_set_patches
    
    combined_patches = [resize(patch, (96,96,3)) for patch in combined_patches]
    
    
    combined_feature_vectors = np.concatenate([segmentation_feature_vectors, support_set_feature_vectors], axis=0)
    
    #labels = np.concatenate([segmentation_feature_vectors_labels, support_set_feature_vectors_labels])
    
    pca = PCA(n_components=100, whiten=True)
    
    pca.fit(combined_feature_vectors)
    
    combined_feature_vectors = pca.transform(combined_feature_vectors)
    
    print(f'Finished extracting pca. Resulting matrix is size {combined_feature_vectors.shape}.')
    
    #labels = np.concatenate([segmentation_feature_vectors_labels, [1]*len(support_set_feature_vectors_labels)])
    
    labels = np.concatenate([segmentation_feature_vectors_labels, support_set_feature_vectors_labels])
    
    optimization_results_object_for_finding_transformation_matrix, initial_transformation_matrix = None, None# run_optimization_to_obtain_prior_transformation_matrix(combined_feature_vectors, labels)


    #nca = NeighborhoodComponentsAnalysis(n_components=2, init=initial_transformation_matrix, verbose=2)
    
    nca = NeighborhoodComponentsAnalysis(n_components=2, verbose=2)

    embedded_feature_vectors = nca.fit_transform(combined_feature_vectors, labels)
    
#     nca = LinearDiscriminantAnalysis(n_components=2)
    
#     nca.fit(combined_feature_vectors, labels)
    
#     embedded_feature_vectors = nca.transform(combined_feature_vectors)

    # embedded_feature_vectors = (initial_transformation_matrix @ combined_feature_vectors.T).T
    
    original_feature_vectors = combined_feature_vectors
    
    background_feature_vectors = nca.transform(pca.transform(segmentation_feature_vectors))
    
    # background_feature_vectors = (initial_transformation_matrix @ segmentation_feature_vectors.T).T
    
    labels_with_support_set_as_one_class = np.concatenate([segmentation_feature_vectors_labels, np.asarray([1]*len(support_set_feature_vectors_labels))])
    
    return embedded_feature_vectors, background_feature_vectors, labels_with_support_set_as_one_class, combined_patches, optimization_results_object_for_finding_transformation_matrix, nca, pca
    
    #return embedded_feature_vectors, original_feature_vectors, labels, combined_patches, None, nca


def find_transformation_matrix(A, x, b, design_matrix_shape):
    '''
    Solve for A

    '''
    A = A.reshape(design_matrix_shape)

    frob_norm = norm((x @ A - b), ord='fro')

    return frob_norm


# def run_optimization_to_obtain_prior_transformation_matrix(original_feature_vectors, labels):
#     '''
#     Express prior on location of design matrix.
    
#     The optimization problem is to find a transformation matrix A such that x@A = b (x and b are row vectors)
    
#     Matrix A transforms original feature vectors (x) to the new (prior) embedding obtained from kernel pca
    
#     b is the (prior) embedding obtained from kernel pca. It captures similarity in labels
    
#     '''
#     #We use kernel pca to give us a prior on where the coordinates would be in the 
#     #embedding based on their similarity in labels
#     X = np.array(labels).reshape(-1,1)

#     pre_computed_kernel = cosine_similarity(X)

#     transformer = KernelPCA(n_components=2, kernel='precomputed')
    
#     #These are the coordinates in the embedding purely based on label similarity
#     X_transformed = transformer.fit_transform(pre_computed_kernel)
    
#     X_transformed = np.vstack([np.array([-1,0]) if int(label) == 0 else np.array([1,0]) for label in labels])

    
#     #We define the parameters of the optimization
#     design_matrix_shape = (original_feature_vectors.shape[1], X_transformed.shape[1])

#     #Reshape design matrix to a one-dim array for scipy.minimize
#     A = rng.random(size=design_matrix_shape).ravel()

#     x = original_feature_vectors

#     b = X_transformed

#     #Enforce constraint to prevent the trivial non zero solution
#     cons = ({'type': 'ineq', 'fun': lambda v:  np.count_nonzero(v) - len(v)})
    
#     #cons = ({'type': 'ineq', 'fun': lambda v:  np.sum(v)})
    
    
    
#     #Loop indefinately until convergence
#     while True:

#         optimization_results_object_for_finding_transformation_matrix = minimize(fun = find_transformation_matrix, x0=A, args=(x, b, design_matrix_shape), options={'disp':True})
        
#         res_x = optimization_results_object_for_finding_transformation_matrix.x

#         x_is_vector_of_zeros = np.allclose(res_x, np.zeros(res_x.shape))
        
#         if optimization_results_object_for_finding_transformation_matrix.success and not(x_is_vector_of_zeros):
            
#             break
    
#     #We get A as a one-dim vector, so we reshape it and transpose it as sklearn expects
#     initial_transformation_matrix = optimization_results_object_for_finding_transformation_matrix.x.reshape(design_matrix_shape).T


#     return optimization_results_object_for_finding_transformation_matrix, initial_transformation_matrix

def run_optimization_to_obtain_prior_transformation_matrix(original_feature_vectors, labels):
    '''
    Express prior on location of design matrix.
    
    The optimization problem is to find a transformation matrix A such that x@A = b (x and b are row vectors)
    
    Matrix A transforms original feature vectors (x) to the new (prior) embedding obtained from kernel pca
    
    b is the (prior) embedding obtained from kernel pca. It captures similarity in labels
    
    '''
    design_matrix_shape = (2, original_feature_vectors.shape[1])
    
    initialization_matrix_A = rng.random(size=design_matrix_shape).ravel()

    #Loop indefinately until convergence
#     while True:

#         optimization_results_object_for_finding_transformation_matrix = minimize(fun = return_objective_function_for_finding_initial_transformation_matrix_for_nca, x0=initialization_matrix_A, args=(original_feature_vectors, labels), options={'disp':True})
        
#         res_x = optimization_results_object_for_finding_transformation_matrix.x

#         x_is_vector_of_zeros = np.allclose(res_x, np.zeros(res_x.shape))
        
#         if optimization_results_object_for_finding_transformation_matrix.success and not(x_is_vector_of_zeros):
            
#             break

    optimization_results_object_for_finding_transformation_matrix = minimize(fun = return_objective_function_for_finding_initial_transformation_matrix_for_nca, x0=initialization_matrix_A, args=(original_feature_vectors, labels), options={'disp':True})
    
    #We get A as a one-dim vector, so we reshape it and transpose it as sklearn expects
    initial_transformation_matrix = optimization_results_object_for_finding_transformation_matrix.x.reshape(design_matrix_shape)


    return optimization_results_object_for_finding_transformation_matrix, initial_transformation_matrix


def return_objective_function_for_finding_initial_transformation_matrix_for_nca(initialization_matrix_A, embeddings, labels):
    '''
    Find embedding matrix using clustering objective
    
    '''
    
    initialization_matrix_A = initialization_matrix_A.reshape(2, embeddings.shape[1])

    transformed_embeddings = initialization_matrix_A @ embeddings.T
    
    clustering_labels = KMeans(n_clusters=len(set(labels))).fit(transformed_embeddings.T).labels_
    
    clustering_metric = homogeneity_score(clustering_labels, labels) - 0.5
    
    return clustering_metric