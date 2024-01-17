# Import the dependencies
#from pyinstrument import Profiler
import sys
import pandas as pd
import numpy as np
import time
import numba
from numba.typed import List

from sklearn.datasets import make_classification
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score

from matplotlib import pyplot as plt

# ## Create a test dataset
X, y = make_classification(n_samples=50, n_features=5, n_informative=2, n_redundant=1, n_repeated=2, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=42) 

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

# Function that defines the clustering algorithm for the subsequent processing
def apply_clustering_algorithm(np_array, clustering_algorithm, params):
    if clustering_algorithm == 'kmedoids':
        model = KMedoids(n_clusters=params.get('n_clusters'),
                         init=params.get('init'),
                         random_state=params.get('random_state'))
    elif clustering_algorithm == 'hdbscan':
        model = HDBSCAN(min_cluster_size=params.get('min_cluster_size'),
                        min_samples=params.get('min_samples'),
                        cluster_selection_method=params.get('cluster_selection_method'),
                        allow_single_cluster=params.get('allow_single_cluster'),
                        n_jobs=params.get('n_jobs'))
    else:
        raise ValueError("Unsupported clustering algorithm")

    model.fit(np_array)
    return model.labels_

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
# Function to accumulate cluster information
def accumulate_cluster_data(clusters, cluster_scatter_separability, cluster_silhouette_score, clusters_list, cluster_scatter_criteria_list, cluster_silhouette_criteria_list):
    """
    Accumulates cluster information for cross-projection normalization.

    Returns:
        A tuple of (clusters_list, cluster_scatter_criteria_list, and cluster_silhouette_criteria_list) for cross-projection normalization.
    """
  
    clusters_list.append(clusters)
    cluster_scatter_criteria_list.append(cluster_scatter_separability)
    cluster_silhouette_criteria_list.append(cluster_silhouette_score)

    return clusters_list, cluster_scatter_criteria_list, cluster_silhouette_criteria_list

        
# =============================================================================
# Function: Normalization of criterion values to remove bias due to dimension - Numba acceleration
@numba.jit(nopython=True)
def cross_projection_normalization(original_data_np, clusters_list, cluster_scatter_criteria_list, cluster_silhouette_criteria_list):
    normalized_values = 0
    n_clusters = len(clusters_list)
    projections = np.zeros((n_clusters, n_clusters))

    for i in range(len(cluster_scatter_criteria_list)):
        for j in range(n_clusters):
            unique_labels_j = np.unique(clusters_list[j])
            for k in range(j+1, n_clusters):  # Only compute for unique pairs j < k
                unique_labels_k = np.unique(clusters_list[k])
                matching_labels = np.intersect1d(unique_labels_j, unique_labels_k)

                # Remove noise labels (-1)
                matching_labels = matching_labels[matching_labels != -1]

                # Initialize an array to hold distances between centroids.
                distances = np.zeros(len(matching_labels))

                for idx, label in enumerate(matching_labels):
                    mask_j = clusters_list[j] == label
                    mask_k = clusters_list[k] == label
                    if mask_j.sum() > 0 and mask_k.sum() > 0:
                        centroid_j = custom_mean(original_data_np[mask_j, :], axis=0)
                        centroid_k = custom_mean(original_data_np[mask_k, :], axis=0)
                        distances[idx] = np.linalg.norm(centroid_j - centroid_k)

                if distances.size > 0:
                    projection = custom_mean(distances)  # compute mean without specifying axis for 1D array
                    projections[j][k] = projection
                    projections[k][j] = projection  # Maintain symmetry

                # Calculate the normalized value for this pair of clusters (j and k)
                normalized_value = (cluster_scatter_criteria_list[i] + cluster_silhouette_criteria_list[i]) / 2 * projections[j][k]
                normalized_values += normalized_value
                    
                    # For debugging, you can print each normalized value
                    #print(f'Normalized value for clusters {j} and {k}: {normalized_value}')

    return normalized_values

# =============================================================================
# Helper function for Sequential Forward Search
def evaluate_feature_subset(current_subset, np_array, clustering_algorithm, clustering_params, clusterslist, cluster_scatter_criteria, cluster_silhouette_criteria):
    t_clusters_list = List()
    t_cluster_scatter_criteria_list = List()
    t_cluster_silhouette_criteria_list = List()
    
    subset_array = np_array[:, current_subset]
    cluster_labels = apply_clustering_algorithm(subset_array, clustering_algorithm, clustering_params)

    scatter_separability = calculate_scatter_separability(subset_array, cluster_labels)
    silhouette_score = calculate_silhouette(subset_array, cluster_labels)

    t_clusters_list, t_cluster_scatter_criteria_list, t_cluster_silhouette_criteria_list = accumulate_cluster_data(
        cluster_labels, scatter_separability, silhouette_score, clusterslist, cluster_scatter_criteria, cluster_silhouette_criteria
    )

    normalized_values = cross_projection_normalization(subset_array, t_clusters_list, t_cluster_scatter_criteria_list, t_cluster_silhouette_criteria_list)

    return normalized_values, current_subset


# =============================================================================
# Sequential Forward Search - Feature Selection (Sequential Execution)
def sequential_forward_selection(np_array, clusters_list, cluster_scatter_criteria_list, cluster_silhouette_criteria_list, clustering_algorithm, clustering_params):
    print(' Sequential Forward Selection ')
    n_features = np_array.shape[1]
    best_features = List()
    best_score = 0
    candidate_features = List(range(n_features))  # Ensure this is a Numba typed List
    best_cluster_labels = None

    while len(candidate_features) > 0:
        feature_scores = List()
        feature_subsets = List()
        
        for feature_idx in candidate_features:
            print(f'(1) Feature Index = {feature_idx}')
            # We will convert the Python list to a Numba typed List before concatenation
            current_subset = List(best_features)
            current_subset.append(feature_idx)
            
            score, subset = evaluate_feature_subset(current_subset, np_array, clustering_algorithm, clustering_params, clusters_list, cluster_scatter_criteria_list, cluster_silhouette_criteria_list)
            feature_scores.append(score)
            print(f'(2) length of feature_scores = {len(feature_scores)}')
            feature_subsets.append(current_subset)  # Store the whole subset as a Numba typed List
        
        best_feature_idx = np.argmax(feature_scores)
        print(f' (3) ++++ Best Feature IDX = {best_feature_idx}')
        
        if feature_scores[best_feature_idx] > best_score:
            print(' (4) Improvement - feature_scores[best_feature_idx] > best_score')
            best_score = feature_scores[best_feature_idx]
            print(f' (5)  ++ Best score = {best_score}')
            best_feature_removal = feature_subsets[best_feature_idx][-1]
            best_cluster_labels = apply_clustering_algorithm(np_array[:, feature_subsets[best_feature_idx]], clustering_algorithm, clustering_params)
            
            temp_feature=None
            temp_feature = feature_subsets[best_feature_idx]
            for itme in temp_feature:
                print(f' (5-1) ITME = {itme}')
                best_features.append(itme)
                
            print(f' (6) length of best_features = {len(best_features)}')
            candidate_features.remove(best_feature_removal)
            print(f' (7)  length of candidate_features = {len(candidate_features)}')
            print(f' (8)   length of clusters list = {len(clusters_list)}')
            
            # Clear the lists to start fresh for the next subset evaluation
            clusters_list.clear()
            cluster_scatter_criteria_list.clear()
            cluster_silhouette_criteria_list.clear()
        else:
            break

    return best_features, best_cluster_labels


# =============================================================================
# Function that evaluates different numbers of clusters to locate the optimal value
def find_optimal_clusters(np_array, clustering_algorithm, clustering_params, max_k, clusters_list, cluster_scatter_criteria, cluster_silhouette_criteria):
    best_k = max_k
    best_criteria_value = -np.inf
    best_features = np.array([])
    best_cluster_labels = None
    t_clusters_list = List.empty_list(numba.float64)
    t_cluster_scatter_criteria_list = List.empty_list(numba.float64)
    t_cluster_silhouette_criteria_list = List.empty_list(numba.float64)

    while best_k > 2:
        print(f' Best K = {best_k}')
        if clustering_algorithm == 'kmedoids':
            clustering_instance = KMedoids(n_clusters=clustering_params.get('n_clusters'),
                         init=clustering_params.get('init'),
                         random_state=clustering_params.get('random_state'))
            current_labels = clustering_instance.fit_predict(np_array)
        elif clustering_algorithm == 'hdbscan':
            clustering_instance = HDBSCAN(min_cluster_size=clustering_params.get('min_cluster_size'),
                        min_samples=clustering_params.get('min_samples'),
                      cluster_selection_method=clustering_params.get('cluster_selection_method'),
                        allow_single_cluster=clustering_params.get('allow_single_cluster'),
                        n_jobs=clustering_params.get('n_jobs'))
            current_labels = clustering_instance.fit_predict(np_array)
        else:
            raise ValueError("Unsupported clustering algorithm")

        scatter_separability = calculate_scatter_separability(np_array, current_labels)
        silhouette_score = calculate_silhouette(np_array, current_labels)
        
        t_clusters_list, t_cluster_scatter_criteria_list, t_cluster_silhouette_criteria_list = accumulate_cluster_data(
            current_labels, scatter_separability, silhouette_score, clusters_list, cluster_scatter_criteria, cluster_silhouette_criteria)

        normalized_criteria = cross_projection_normalization(np_array, t_clusters_list, t_cluster_scatter_criteria_list, t_cluster_silhouette_criteria_list)

        if normalized_criteria > best_criteria_value:
            best_criteria_value = normalized_criteria
            best_cluster_labels = current_labels
            best_features, _ = sequential_forward_selection(
                np_array, t_clusters_list, t_cluster_scatter_criteria_list, t_cluster_silhouette_criteria_list, clustering_algorithm, clustering_params
            )
        else:
            best_k -= 1


    return best_k, best_features, best_cluster_labels

# =============================================================================
# Main Function
def feature_selection_and_clustering(np_array, clustering_algorithm, max_clusters=145):
    print('Entered feature_selection_and_clustering ')
    
    clustering_params = dict()
    # Initialize global variables for accumulation
    clusters_list = List()
    cluster_scatter_criteria_list = List.empty_list(numba.float64)
    cluster_silhouette_criteria_list = List.empty_list(numba.float64)
    
    if clustering_algorithm == 'kmedoids':
        clustering_params = {'n_clusters': max_clusters, 'init': 'k-medoids++', 'random_state': 42}
    elif clustering_algorithm == 'hdbscan':
        clustering_params = {'min_cluster_size': 5, 'min_samples': None, 'cluster_selection_method': 'eom', 'allow_single_cluster': 'False', 'n_jobs': -1}
    else:
        raise ValueError("Unsupported clustering algorithm")

    best_k, best_features, best_labels = find_optimal_clusters(np_array, clustering_algorithm, clustering_params, max_clusters, clusters_list, cluster_scatter_criteria_list, cluster_silhouette_criteria_list)
    # Output the results
    print(' ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ')
    print(f'Optimal number of clusters: {best_k}')
    print(f'Optimal number of features: {len(best_features)}')
    print(f'Cluster label distribution: {np.unique(best_labels, return_counts=True)}')
    print(' ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ')

    return best_k, best_features, best_labels

# =============================================================================
# ### Perform PFA

# #### KMedoids
# Start timing
start = time.perf_counter()

best_k = -99
best_kmedoid_features = []

# Run the experiment using the complete (non-pca) dataframe and identify the clustering algorithm by name.
best_k, best_kmedoid_features, best_kmedoid_labels = feature_selection_and_clustering(test_np, 'kmedoids')

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
