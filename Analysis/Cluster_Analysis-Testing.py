# Import the dependencies
#from pyinstrument import Profiler
import sys
import pandas as pd
import numpy as np
import time
import numba
import random
from numba.typed import List

from sklearn.datasets import make_classification
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score

from matplotlib import pyplot as plt

# ## Create a test dataset
X, y = make_classification(n_samples=175, n_features=14, n_informative=4, n_redundant=4, n_repeated=6, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=2.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=42) 

plt.scatter(X[:,0], X[:,1])
plt.show()

test_df = pd.DataFrame(X)
test_df = test_df.add_prefix('feature_')
test_df

# Convert the dataframe to a Numpy array for enhanced performance
test_np = test_df.to_numpy()

# ## Principal Feature Analysis ##

# #### Define functions to select dataset features that provide relevant information for clustering. 
# ##### Only important features are used to compute clusters from the complete (non-pca) dataset.

#profiler = Profiler()
#profiler.start()


# =============================================================================
# Custom processing function to override limitations of Numba compatibility with Numpy features
@numba.jit(nopython=True)
def custom_mean(arr, axis=0):
    print('    Entered custom_mean ')
    if arr.ndim == 1:
        return arr.sum() / arr.shape[0]
    elif arr.ndim == 2:
        if axis == 0:
            return arr.sum(axis=0) / arr.shape[0]
        elif axis == 1:
            return arr.sum(axis=1) / arr.shape[1]
    raise ValueError("custom_mean function received an array that it can't handle with axis = {}")

# =============================================================================
# Function: Silhouette Coefficient Calculation
def calculate_silhouette(np_array, labels):
    print('   Entered calculate_silhouette ')
    if len(np.unique(labels)) > 1:
        silhouette_val = silhouette_score(np_array, labels)
        return silhouette_val
    else:
        return 0

# =============================================================================
# Function: Scatter Separability Calculation
def calculate_scatter_separability(np_array, labels):
    print('   Entered calculate_scatter_separability ')
    unique_labels = np.unique(labels)
    n_features = np_array.shape[1]
    overall_mean = custom_mean(np_array, axis=0)
    
    S_w = np.zeros((n_features, n_features))
    S_b = np.zeros((n_features, n_features))

    for label in unique_labels:
        X_k = np_array[labels == label]
        mean_k = custom_mean(X_k, axis=0).reshape(n_features, 1)
        diff = X_k - mean_k.T
        S_w += np.dot(diff.T, diff)
        mean_diff = mean_k - overall_mean.reshape(n_features, 1)
        S_b += X_k.shape[0] * np.dot(mean_diff, mean_diff.T)

    # Check if S_w is invertible
    if np.linalg.cond(S_w) < 1/sys.float_info.epsilon:
        final_ssc = np.trace(np.linalg.inv(S_w).dot(S_b))
    else:
        final_ssc = 0

    return final_ssc



# =============================================================================
# Function: Normalization of criterion values to remove bias due to number of clusters - Numba acceleration
@numba.jit(nopython=True)
def cross_projection_normalization(original_data_np, cluster_labels, scatter_criteria_score, silhouette_criteria_score):
    print('   Entered cross_projection_normalization ')
    normalized_score = 0
    unique_labels = np.unique(cluster_labels)
    for uni_lab in unique_labels:
        print(f'    CPN - Unique Labels = {uni_lab}')
    n_clusters = len(unique_labels)
    print(f'    CPN - Number of Clusters - {n_clusters}')
    projections = np.zeros((n_clusters, n_clusters))

    for j in range(n_clusters):
        for k in range(j+1, n_clusters):
            matching_labels = np.intersect1d(unique_labels[j], unique_labels[k])
            matching_labels = matching_labels[matching_labels != -1] # Remove noise labels (-1)

            distances = np.zeros(len(matching_labels))

            for idx, label in enumerate(matching_labels):
                print('    CPN -- Entered idx, label for-loop ')
                mask_j = cluster_labels == label  # Mask original data_np directly
                mask_k = cluster_labels == label
                print(f'    CPN ---- mask_j.sum = {mask_j.sum()}   mask_k.sum = {mask_k.sum()} +_+_+_+_==========')
                if mask_j.sum() > 0 and mask_k.sum() > 0:
                    centroid_j = custom_mean(original_data_np[mask_j, :], axis=0)
                    centroid_k = custom_mean(original_data_np[mask_k, :], axis=0)
                    distances[idx] = np.linalg.norm(centroid_j - centroid_k)

            print(f'    CPN - Distance Size = {distances.size}')
            if distances.size > 0:
                projection = custom_mean(distances)
                print(f'    CPN - Projection = {projection}')
                projections[j][k] = projection
                projections[k][j] = projection

                normalized_value = (scatter_criteria_score + silhouette_criteria_score) / 2 * projections[j][k]
                normalized_score += normalized_value
            else:
                print('    CPN -- Entered final else-clause to compute normalized score ')
                normalized_value = (scatter_criteria_score + silhouette_criteria_score) / 2
                normalized_score += normalized_value

    return normalized_score

# =============================================================================
# Helper function for Sequential Forward Search
def evaluate_feature_subset(subset_array, np_array, cluster_labels):
    print('  Entered evaluate_feature_subset ')
    scatter_separability = calculate_scatter_separability(subset_array, cluster_labels)
    silhouette_score = calculate_silhouette(subset_array, cluster_labels)
    normalized_score = cross_projection_normalization(subset_array, cluster_labels, scatter_separability, silhouette_score)

    return normalized_score

# =============================================================================
# Function that evaluates different numbers of clusters to locate the optimal value
def optimal_feature_clusters(np_array, clustering_algorithm):
    print(' Entered optimal_feature_clusters ')
    np_array_feature_indices = np_array.shape[1]
    available_indices = List(range(np_array_feature_indices))  # Initial list of available indices
    temp_random_indices = available_indices.copy()
    initial_k = np.array([2, 3, 4, 5])
    selected_features = List()
    random.seed(42)

    evaluate = True
    first_pass = True
    init_k = 2
    best_k = init_k
    best_score = -np.inf
    best_feature_index = None

    while evaluate:     # Simple test to enable and continue evaluation.
        print(f' Best K = {init_k}')

        if clustering_algorithm == 'kmedoids':
            clustering_instance = KMedoids(n_clusters=init_k, init='k-medoids++', max_iter=500, random_state=42)          
        elif clustering_algorithm == 'hdbscan':
            clustering_instance = HDBSCAN(min_cluster_size=10, min_samples=20, cluster_selection_method='eom', allow_single_cluster='True', n_jobs=-1)
        else:
            raise ValueError("Unsupported clustering algorithm")

        while first_pass:
            # Select the best feature to start the evaluation
            while temp_random_indices:
                random_idx = random.choice(temp_random_indices)
                # Process each individual feature
                subset_array = np_array[:, [random_idx]]
                current_labels = clustering_instance.fit_predict(subset_array)
                # Score the feature                  
                normalized_score = evaluate_feature_subset(subset_array, np_array, current_labels)
        
                # Update best feature if necessary
                if normalized_score > best_score:
                    best_score = normalized_score
                    best_feature_index = random_idx
                
                temp_random_indices.remove(random_idx)
    
            # Add best feature to selected features and remove its index from available indices
            selected_features.append(best_feature_index)
            available_indices.remove(best_feature_index)
            best_combination_score = best_score
            first_pass = False # Set the flag to prevent a rerun of the initial feature selection.
            
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Initial feature identified - continuing with combination evaluation ^^^^^^^^^^^^^^^^^^^^^^^^')
        
        for init_k in initial_k:
            if clustering_algorithm == 'kmedoids':
                print(f'****** k-value set to {init_k} ***************************')
                clustering_instance = KMedoids(n_clusters=init_k, init='k-medoids++', random_state=42)  
            # Process combinations with the current selected features

            best_add = None
            
            for i in available_indices:
                # Create combination subset
                print(f' Evaluate combined features - current feature = {i} ')
                combination_array = np.hstack([np_array[:, [i]], np_array[:, selected_features]])
                current_labels = clustering_instance.fit_predict(combination_array)
    
                # Score the combination            
                normalized_score = evaluate_feature_subset(combination_array, np_array, current_labels)
                print(f'*** Normalized score = {normalized_score}   ::::   Best Score = {best_combination_score} ***')
    
                # Update best combination if necessary
                if normalized_score > best_combination_score:
                    best_combination_score = normalized_score
                    print(f'  ++++++++ Best Combination Score Updated = {best_combination_score}    ++++++++')
                    best_add = i
                    best_k = init_k
    
            # If a better combination was found, add its feature to selected features
            if best_add is not None:
                selected_features.append(best_add)
                available_indices.remove(best_add)
            else:
                print('xxxxx  No further progress >>>> Moving to next K >>>>>>>')

        evaluate = False

    return best_k, selected_features


# =============================================================================
# Main Function
def feature_selection_and_clustering(np_array, clustering_algorithm, max_clusters=145):
    print('Entered feature_selection_and_clustering ')
    
    best_k, best_features = optimal_feature_clusters(np_array, clustering_algorithm)

    return best_k, best_features

# =============================================================================
# ### Perform PFA

# #### KMedoids
# Start timing
start = time.perf_counter()

best_kmedoid_features = []

# Run the experiment using the complete (non-pca) dataframe and identify the clustering algorithm by name.
best_k, best_kmedoid_features = feature_selection_and_clustering(test_np, 'kmedoids')

print(best_kmedoid_features)
# Stop timing
stop = time.perf_counter()

print(' ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ')
print(f' ^^^ PFA KMedoids Clustering Execution in {stop - start:0.4f} seconds ^^^ ')
print(' ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ')

#profiler.stop()

#profiler.write_html("results_file.html", show_all=False, )


# #### HDBSCAN
# Start timing
# =============================================================================
# start = time.perf_counter()
# 
# best_k = -99
# best_hdbscan_features = []
# # Run the experiment using the complete (non-pca) dataframe
# not_used, best_hdbscan_features = feature_selection_and_clustering(test_df, 3,'hdbscan')
# 
# # Stop timing
# stop = time.perf_counter()
# 
# print(' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ')
# print(f' +++ PFA HDBSCAN Clustering Execution in {stop - start:0.4f} seconds +++ ')
# print(' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ')
# 
# =============================================================================
