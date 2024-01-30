# Import the dependencies
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
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score

from matplotlib import pyplot as plt

# ## Create a test dataset
X, y = make_classification(n_samples=5000, n_features=50, n_informative=12, n_redundant=38, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=6.06109, hypercube=True, shift=0.09384, scale=.345868973, shuffle=True, random_state=42) 

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

# =============================================================================
# Custom processing function to override limitations of Numba compatibility with Numpy features
@numba.jit(nopython=True)
def custom_mean(arr, axis=0):
    if arr.ndim == 1:
        return arr.sum() / arr.shape[0]
    elif arr.ndim == 2:
        if axis == 0:
            return arr.sum(axis=0) / arr.shape[0]
        elif axis == 1:
            return arr.sum(axis=1) / arr.shape[1]
    raise ValueError("custom_mean function received an array that it can't handle with axis = {}")

# =============================================================================
# Function: Calinski Harbasz Score Calculation
def calculate_calinski_harbasz(np_array, labels):
    if len(np.unique(labels)) > 1:
        calinski_harbasz = calinski_harabasz_score(np_array, labels)
        return calinski_harbasz
    else:
        return 0

# =============================================================================
# Function: Davies-Bouldin Score Calculation
def calculate_davies_bouldin(np_array, labels):
    if len(np.unique(labels)) > 1:
        davies_bouldin = davies_bouldin_score(np_array, labels)
        return davies_bouldin
    else:
        return 0

# =============================================================================
# Function: Silhouette Coefficient Calculation
def calculate_silhouette(np_array, labels):
    if len(np.unique(labels)) > 1:
        silhouette_val = silhouette_score(np_array, labels)
        return silhouette_val
    else:
        return 0

# =============================================================================
# Function: Scatter Separability Calculation
def calculate_scatter_separability(np_array, labels):
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
def cross_projection_normalization(clustering_medoids, scatter_criteria_score, silhouette_criteria_score, davies_bouldin_score, calinski_harbasz_index):
    print('   Entered cross_projection_normalization ')
    n_clusters = len(clustering_medoids)
    projections = np.zeros((n_clusters, n_clusters))

    for j in range(n_clusters):
        for k in range(j + 1, n_clusters):
            medoid_j = clustering_medoids[j]
            medoid_k = clustering_medoids[k]
            distance = np.linalg.norm(medoid_j - medoid_k)
            projections[j][k] = distance
            projections[k][j] = distance

    # Flatten the array and filter non-zero distances then calculate the mean
    flat_projections = projections.ravel()
    non_zero_projections = flat_projections[flat_projections > 0]
    mean_projection = np.mean(non_zero_projections)

    # Normalizing the criteria scores with the mean of projections
    # Adjusting the formula to consider Davies-Bouldin Score. Recall: For Davies-Bouldin, lower is better.
    # We add 1 to the Davies-Bouldin score to ensure it doesn't lead to division by zero or negative values.

    # Combined normalization factor incorporates all metrics.
    normalization_factor = (1 + mean_projection + davies_bouldin_score) 

    normalized_score = (scatter_criteria_score + silhouette_criteria_score + calinski_harbasz_index) / normalization_factor

    return normalized_score

# =============================================================================
# Helper function for Sequential Forward Search
def evaluate_feature_subset(subset_array, np_array, cluster_labels, clustering_medoids):
    scatter_separability = calculate_scatter_separability(subset_array, cluster_labels)
    silhouette_score = calculate_silhouette(subset_array, cluster_labels)
    davies_bouldin_score = calculate_davies_bouldin(subset_array, cluster_labels)
    calinski_harbasz_index = calculate_calinski_harbasz(subset_array, cluster_labels)
    normalized_score = cross_projection_normalization(clustering_medoids, scatter_separability, silhouette_score, davies_bouldin_score, calinski_harbasz_index)

    return normalized_score

# =============================================================================
# Function that evaluates different numbers of clusters to locate the optimal value
def optimal_feature_clusters(np_array, clustering_algorithm):
    np_array_feature_indices = np_array.shape[1]
    available_indices = List(range(np_array_feature_indices))  # Initial list of available indices
    n_features = len(available_indices)
    initial_k = np.array([2, 3, 4, 6, 7, 8, 9, 10])
    interim_features = []
    random.seed(42)
    evaluate = True
    init_k = 2
    best_k = init_k
    best_score = 0
    processed_features = 0

    while evaluate:     # Simple test to enable and continue evaluation.
        print(f' Best K = {init_k}')

        if clustering_algorithm == 'kmedoids':
            clustering_instance = KMedoids(n_clusters=init_k, init='k-medoids++', metric='manhattan', random_state=42)          
        elif clustering_algorithm == 'hdbscan':
            clustering_instance = HDBSCAN(min_cluster_size=10, min_samples=20, cluster_selection_method='eom', store_centers="medoid", allow_single_cluster='True', n_jobs=-1)
        else:
            raise ValueError("Unsupported clustering algorithm")

        while processed_features < 0.8 * n_features:
                starter_set = np.random.choice(n_features, size=max(1, int(0.1 * n_features)))
                best_feature = None
                best_score = -np.inf
        
                for feature in range(n_features):
                    if feature not in starter_set and feature not in interim_features:  # Check for both conditions
                        combined_features = np.concatenate([starter_set, [feature]])
                        subset_array = np.hstack([np_array[:, combined_features]])
                        current_labels = clustering_instance.fit_predict(subset_array)
                        clustering_medoids = clustering_instance.cluster_centers_
                        # Score the feature                  
                        normalized_score = evaluate_feature_subset(subset_array, np_array, current_labels, clustering_medoids)
        
                        # Update best feature if necessary
                        if normalized_score > best_score:
                            best_score = normalized_score
                            best_feature = feature
        
                # Ensure best_feature is not already in interim_features before appending
                if best_feature is not None and best_feature not in interim_features:
                    interim_features.append(best_feature)
                    
                processed_features += len(starter_set)  # Account for multiple features in starter set
                print(f' Processed Features = {processed_features}')
        best_combination_score = best_score
                
            
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Initial features identified - continuing with combination evaluation ^^^^^^^^^^^^^^^^^^^^^^^^')
        
        for init_k in initial_k:
            if clustering_algorithm == 'kmedoids':
                print(f'****** k-value set to {init_k} ***************************')
                clustering_instance = KMedoids(n_clusters=init_k, init='k-medoids++', metric='manhattan', random_state=42)  
            # Process combinations with the current selected features

            best_add = None
            
            for i in available_indices:
                # Create combination subset
                print(f' Evaluate combined features - current feature = {i} ')
                combination_array = np.hstack([np_array[:, [i]], np_array[:, interim_features]])
                current_labels = clustering_instance.fit_predict(combination_array)
                clustering_medoids = clustering_instance.cluster_centers_
    
                # Score the combination            
                normalized_score = evaluate_feature_subset(combination_array, np_array, current_labels, clustering_medoids)
                print(f'*** Normalized score = {normalized_score}   ::::   Best Score = {best_combination_score} ***')
    
                # Update best combination if necessary
                if normalized_score > best_combination_score:
                    best_combination_score = normalized_score
                    print(f'  ++++++++ Best Combination Score Updated = {best_combination_score}    ++++++++')
                    best_add = i
                    best_k = init_k
    
            # If a better combination was found, add its feature to selected features
            if best_add is not None:
                interim_features.append(best_add)
                available_indices.remove(best_add)
            else:
                print('xxxxx  No further progress >>>> Moving to next K >>>>>>>')

        evaluate = False
        print(' Processing completed ')

    return best_k, interim_features


# =============================================================================
# Main Function
def feature_selection_and_clustering(np_array, clustering_algorithm):
    best_k, best_features = optimal_feature_clusters(np_array, clustering_algorithm)

    return best_k, best_features

# =============================================================================
# ### Perform PFA

# #### KMedoids
# Start timing
start = time.perf_counter()

# Run the experiment using the complete (non-pca) dataframe and identify the clustering algorithm by name.
best_k, best_kmedoid_features = feature_selection_and_clustering(test_np, 'kmedoids')

print(f' Best k = {best_k}')
print(f' Best features = {best_kmedoid_features}')
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
