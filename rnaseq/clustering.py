import numpy as np
import tkinter as tk
from . import DrawApp
import seaborn as sb
from sklearn.base import clone
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import fastcluster as fc
from scipy.cluster.hierarchy import fcluster, dendrogram, set_link_color_palette
from biosppy.clustering import consensus_kmeans, _life_time
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

#%% Hierarchical Clustering

# Define what is going to be clustered
def what_to_cluster(data_frame, cluster_by):
    
    """
    Define the data frame to use for clustering based on the cluster_by parameter.

    Parameters:
        data_frame: A pandas DataFrame object containing the data to be clustered.
        cluster_by: A string specifying whether to cluster by 'gene' or 'sample'.

    Returns:
        data_cluster: A pandas DataFrame object representing the data frame to be used for clustering.
    """
    if cluster_by == 'gene':
        data_cluster = data_frame
        # If cluster_by is 'gene', the input data_frame is assigned directly to the data_cluster variable.
        
    elif cluster_by == 'sample':
        data_cluster = data_frame.transpose(copy=True)
        # If cluster_by is 'sample', the data_frame is transposed (rows become columns) and the resulting transposed data frame is assigned to the data_cluster variable.

    else:
        print ('error = cluster_by must be gene or sample')
        # If cluster_by has any other value, an error message is printed.
    
    return data_cluster

# Hierarchical Clustering with a desired method and metric
def hierarchical_matrix(data_frame, method, metric):
    """
    Performs hierarchical clustering on a given data frame using the specified parameters.
    
    Args:
        data_frame (pandas.DataFrame): The input data frame containing the data to be clustered.
        method: A string parameter specifying the linkage method to be used for clustering. 
                The method options are 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'
        metric: A string parameter specifying the distance metric to be used for clustering.
                The metric options are 'euclidean', 'cosine', 'correlation'
    Returns:
        The resulting hierarchical clustering linkage matrix.
    """
    
    #preserve_input= false, usa menos memoria
    
    linkage_matrix = fc.linkage(data_frame, method=method, metric=metric, preserve_input=True)
            
    return linkage_matrix


# Hierarchical Clustering of all methods and metrics

def hierarchical_matrix_all(data_frame, methods, metrics):
    """
    Performs hierarchical clustering on a given data frame using the specified parameters.
    
    Parameters:
        data_frame (pandas.DataFrame): The input data frame containing the data to be clustered.
        methods (list): A list of string parameters specifying the linkage methods to be used for clustering.
                        Available method options are 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'.
        metrics (list): A list of string parameters specifying the distance metrics to be used for clustering.
                        Available metric options are 'euclidean', 'cosine', 'correlation'.
                        
    Returns:
        A dictionary containing the resulting hierarchical clustering linkage matrices, with keys representing
        the combination of method and metric used.
    """
    linkage_matrices = {}
    
    for method in methods:
        for metric in metrics:
            if (method == 'centroid' or method == 'median' or method == 'ward') and metric != 'euclidean':
                # Skip combination if method is 'centroid', 'median', or 'ward' but metric is not 'euclidean'
                continue
            linkage_matrix = fc.linkage(data_frame, method=method, metric=metric, preserve_input=True)
            key = f"{method}_{metric}"
            linkage_matrices[key] = linkage_matrix
    
    return linkage_matrices

def hierarchical_partition_labels(linkage_matrix, criterion, n):
    """
    Performs hierarchical partitioning on a given linkage matrix.

    Parameters:
        linkage_matrix (ndarray): The linkage matrix obtained from hierarchical clustering.
        criterion (str): The criterion for partitioning. It can be either 'maxcluster' or 'life_time'.
        n (float): The threshold parameter for partitioning. Only required for the 'maxcluster' criterion.

    Returns:
        labels: An array of labels indicating the cluster assignments for each sample.
    """
    
    if  criterion == 'maxcluster':
        labels = fcluster(linkage_matrix, n, criterion='maxclust')
        
    if  criterion == 'life_time':
        N = (len(linkage_matrix)+1)
        labels = _life_time(linkage_matrix, N)
    
    return labels


def hierarchical_partition_threshold(linkage_matrix, n):
    """
    Performs hierarchical partitioning on a given linkage matrix.

    Parameters:
        linkage_matrix (ndarray): The linkage matrix obtained from hierarchical clustering.
        n (float): Number of clusters to form or life time if 0.

    Returns:
        threshold: Float that corresponds to the threshold to form n number of clusters.
    """
    
    if  n == 0:
        df = np.diff(linkage_matrix[:, 2])
        # find maximum difference
        idx_max = np.argmax(df)
        # find threshold
        th = ((linkage_matrix[idx_max, 2]+linkage_matrix[idx_max+1,2])/2)
        
    else  :
        th=((linkage_matrix[-n,2]+linkage_matrix[-(n-1),2])/2)
    
    return th


def hierarchical_partition(linkage_matrix, criterion, n):
    """
    Performs hierarchical partitioning on a given linkage matrix.

    Parameters:
        linkage_matrix (ndarray): The linkage matrix obtained from hierarchical clustering.
        criterion (str): The criterion for partitioning. It can be either 'maxcluster' or 'life_time'.
        n (float): Number of clusters to form or life time if 0.

    Returns:
        threshold: Float that corresponds to the threshold to form n number of clusters.
        labels: An array of labels indicating the cluster assignments for each sample.
    """
    if  criterion == 'maxcluster':
        th=((linkage_matrix[-n,2]+linkage_matrix[-(n-1),2])/2)
        labels = fcluster(linkage_matrix, n, criterion='maxclust')
        
    if  criterion == 'life_time':
        N = (len(linkage_matrix)+1)

        if N < 3:
            raise ValueError("The number of objects N must be greater then 2.")

        # compute differences from Z distances
        df = np.diff(linkage_matrix[:, 2])
        # find maximum difference
        idx_max = np.argmax(df)
        mx_dif = df[idx_max]
        # find minimum difference
        mi_dif = np.min(df[np.nonzero(df != 0)])

            
        # find threshold link distance
        th_link = linkage_matrix[idx_max, 2]
        # links above threshold
        idxs = linkage_matrix[np.nonzero(linkage_matrix[:, 2] > th_link)[0], 2]
        #number of links above threshold +1 = number of clusters and singletons
        cont = len(idxs) + 1

        # condition (perceber melhor)
        if mi_dif != mx_dif:
            if mx_dif < 2 * mi_dif:
                cont = 1

        if cont > 1:
            labels = fcluster(linkage_matrix, cont, 'maxclust')
        else:
            labels = np.arange(N, dtype='int')

        
        th = ((linkage_matrix[idx_max, 2]+linkage_matrix[idx_max+1,2])/2)
    
    return {'threshold':th, 'samples_labels':labels}


#%% CLUSTERS EVALUATION

# Evaluate the performed clusters

def evaluate_clustering(data_frame, labels):
    """
    Evaluate the quality of hierarchical clustering results using multiple evaluation metrics.

    Parameters:
        data_frame: A pandas DataFrame object containing the data used for clustering.
        linkage_matrix: The resulting hierarchical clustering linkage matrix.

    Returns:
        A dictionary containing the evaluation scores:
        - 'calinski_harabasz': The Calinski-Harabasz Index.
        - 'davies_bouldin': The Davies-Bouldin Index.
        - 'silhouette': The Silhouette Coefficient.
    """

    # Calculate evaluation scores
    scores = {}

    # Calinski-Harabasz Index
    ch_score = calinski_harabasz_score(data_frame, labels)
    scores['calinski_harabasz'] = ch_score

    # Davies-Bouldin Index
    db_score = davies_bouldin_score(data_frame, labels)
    scores['davies_bouldin'] = db_score

    # Silhouette Coefficient
    silhouette_avg = silhouette_score(data_frame, labels)
    scores['silhouette'] = silhouette_avg

    return scores


#%% PLOT HIERARCHICAL CLUSTERING

# plot the hierarchical clustering

def plot_dendrogram(linkage_matrix, labels, cluster_threshold, cmap):
    """
   Plot a Dendrogram
    
    This function generates and displays a dendrogram plot based on the provided linkage matrix,
    which represents the hierarchical clustering of data points. The dendrogram illustrates the
    hierarchical structure of clusters in the data, with vertical lines indicating cluster
    mergers at different levels of similarity.
    
    Parameters:
    - linkage_matrix (array-like): The linkage matrix resulting from hierarchical clustering,
      defining how clusters are merged.
    - labels (list or array-like): Labels or identifiers for the data points being clustered.
    - cluster_threshold (float): A threshold value to color clusters above it differently,
      aiding in the identification of meaningful clusters.
    - cmap (str or colormap, optional): The colormap used for coloring clusters.
    
    Returns:
    - None: The function displays the dendrogram plot but does not return any values.
    
    """
    plt.figure(figsize=(20,20))
    plt.title("Dendrogram")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    sbcmap = sb.color_palette(cmap, n_colors=len(set(labels)))
    set_link_color_palette([colors.rgb2hex(rgb[:3]) for rgb in sbcmap])
    dendrogram(Z=linkage_matrix, color_threshold=cluster_threshold, labels=labels, above_threshold_color='#b3b3b3')
    
    return plt.show()



#%% BI-CLUSTERIRNG

import seaborn as sb

def bicluster(data, method, metric, figsize=(20, 20), dendrogram_ratio=(0.2, 0.2), cmap='Spectral_r'):
    """
    Performs biclustering on the given data using the specified parameters and visualizes the results.

    Parameters:
        data (pandas.DataFrame or numpy.ndarray): The input data to be biclustered.
        method (str): The linkage method to be used for clustering. Available options are:
                      - 'single', complete','average','weighted','centroid','median','ward'.
        metric (str): The distance metric to be used for clustering. Available options are:
                      - 'euclidean','cosine','correlation'
        figsize (tuple, optional): The figure size for the resulting clustermap. Defaults to (20, 20).
        dendrogram_ratio (tuple, optional): The ratio of the dendrogram sizes. Defaults to (0.2, 0.2).
        cmap (str or colormap, optional): The colormap to be used for the resulting clustermap. Defaults to 'Spectral_r'.

    Returns:
        seaborn.matrix.ClusterGrid: The resulting clustermap object.
    """
    sb.set_theme(color_codes=True)
    bicluster_grid = sb.clustermap(data, method=method, metric=metric, 
                                   figsize=figsize, dendrogram_ratio=dendrogram_ratio, cmap=cmap)
    return bicluster_grid

#%% 9. K-Testing

# Cluster Stability using bootstrapping (Adjusted Rand Index)
def cluster_stability(X, est, n_iter=20):
    labels = []
    indices = []
    for i in range(n_iter):
        # draw bootstrap samples, store indices
        sample_indices = np.random.randint(0, X.shape[0], X.shape[0])
        indices.append(sample_indices)
        est = clone(est)
        if hasattr(est, "random_state"):
            # randomize estimator if possible
            est.random_state = np.random.randint(1e5)
        X_bootstrap = X[sample_indices]
        est.fit(X_bootstrap)
        # store clustering outcome using original indices
        relabel = -np.ones(X.shape[0], dtype=int)
        relabel[sample_indices] = est.labels_
        labels.append(relabel)
    scores = []
    for l, i in zip(labels, indices):
        for k, j in zip(labels, indices):
            # we also compute the diagonal which is a bit silly
            in_both = np.intersect1d(i, j)
            scores.append(adjusted_rand_score(l[in_both], k[in_both]))
    return np.mean(scores)

def k_plot(metagene_map, n_range):
    """
    Plot of average stability and silhouette scores for K values.
    Parameters:
        n_range: Maximum K value.
    Returns:
        Line plot.
    """
    stability = []
    silhouette = []
    davies = []
    cluster_range = range(2, n_range, 1)
    
    for n_clusters in cluster_range:
        km = KMeans(n_clusters=n_clusters, n_init='auto')
        stability.append(cluster_stability(metagene_map, km))
        
        silhouette_scores = []
        davies_scores = []
        
        for _ in range(20): 
            kmeans = km.fit_predict(metagene_map)
            sil = silhouette_score(metagene_map, kmeans)
            silhouette_scores.append(sil)
            dav = davies_bouldin_score(metagene_map, kmeans)
            davies_scores.append(dav)
        
        davies.append(np.mean(davies_scores))
        silhouette.append(np.mean(silhouette_scores))
    
    fig, ax1 = plt.subplots()

    ax1.plot(cluster_range, stability, label="Stability", color='k')
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Stability", color='k')

    ax2 = ax1.twinx()
    ax2.plot(cluster_range, silhouette, label="Silhouette", color='b')
    ax2.set_ylabel("Silhouette", color='k')

    ax1_lines, ax1_labels = ax1.get_legend_handles_labels()
    ax2_lines, ax2_labels = ax2.get_legend_handles_labels()

    ax1.legend(ax1_lines + ax2_lines, ax1_labels + ax2_labels, loc="upper right")

    plt.title("Stability and Silhouette Scores vs. Number of Clusters (K)")
    plt.xticks(cluster_range)
    plt.show()

#%% 10. KMeans

def KMeans_clustering(main_map, metagene_map, n_clusters):
    """
    Performs the KMeans algorithm on the metagenes and creates a plot for easy cluster visualization with a 
    custom colormap. Does not support more than 20 clusters. Also returns a barplot of each cluster's average silhouette score.
    Parameters:
        n_clusters (int): number of clusters
    Returns:
        cluster_labels_grid (list): list of shape (map_size, map_size) where each entry is the cluster to which the node in the sam eposition belongs to.
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42, verbose=0,init="k-means++").fit(metagene_map)
    cluster_labels = kmeans.labels_
    cluster_labels_grid = cluster_labels.reshape((main_map.map_size, main_map.map_size))
    custom_colormap=['#9e0142','#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#d9ef8b', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2', '#ffffff', '#878787', '#1a1a1a', '#c51b7d', '#b2abd2', '#4d9221', '#35978f', '#313695', '#8c510a']
    cmap = colors.ListedColormap(custom_colormap[:n_clusters])
    plt.matshow((cluster_labels_grid), cmap=cmap, origin='lower')
    plt.title('KMeans Clustering of Metagenes')
    plt.colorbar(label='Cluster Label').set_ticks(np.arange(0,n_clusters,1))
    plt.show()

    clusters_of_interest = range(0,n_clusters)
    mask = [label in clusters_of_interest for label in cluster_labels]
    X_selected = metagene_map[mask]
    labels_selected = np.array([label for label in cluster_labels if label in clusters_of_interest])
    silhouette_avg = silhouette_score(X_selected, labels_selected)
    silhouette_vals = silhouette_samples(X_selected, labels_selected)
    cluster_silhouette_vals = {}

    for k in clusters_of_interest:
        cluster_mask = (labels_selected == k)
        cluster_vals = silhouette_vals[cluster_mask]
        cluster_size = cluster_mask.sum()
        if cluster_size > 1: 
            cluster_silhouette_vals[k] = cluster_vals
        else:
            cluster_silhouette_vals[k] = np.array([]) 

    average_silhouette_scores = {k: (np.mean(v) if len(v) > 0 else np.nan) for k, v in cluster_silhouette_vals.items()}
    print(f"Average Silhouette Score within Cluster: {average_silhouette_scores}")
    print(f"Total Average: {silhouette_avg}")
    keys=[]
    items=[]
    for key in (average_silhouette_scores):
        keys.append(key)
        items.append(average_silhouette_scores[key])
    plt.bar(keys,items)
    plt.axhline(0.25, color='red', linestyle='--')
    plt.xlabel('Cluster Label')
    plt.ylabel('Silhouette Score')
    plt.xticks(range(0,n_clusters,1))
    plt.show
    return cluster_labels_grid

def clustered_symbols_dict(cluster_labels_grid, map_size, genesymbol_grid):
    """
    Builds dictionary genegrid, where each key is a cluster and corresponding to it is every gene within it, by symbol.
    Parameters:
        cluster_labels_grid (list): list of shape (map_size, map_size) where each entry is the cluster to which the node in the sam eposition belongs to.
        map_size (int): Length of the edges of squared SOM. Map size is actually (map_size*map_size).
    Returns:
        clustered_genes_names (dictionary): dictionary with every cluster and its corresponding gene symbols.
    """
    clustered_genes_symbols={}
    for y in range(map_size):
        for x in range(map_size):
            try:
                if cluster_labels_grid[y][x] in clustered_genes_symbols:
                    clustered_genes_symbols[cluster_labels_grid[y][x]].extend(genesymbol_grid[(x, y)])
                else:
                    clustered_genes_symbols[cluster_labels_grid[y][x]] = list(genesymbol_grid[(x, y)])
            except KeyError:
                continue
    return clustered_genes_symbols

def clustered_ids_dict(cluster_labels_grid, map_size, geneid_grid):
    """
    Builds dictionary genegrid, where each key is a cluster and corresponding to it is every gene within it, by EnsemblID.
    Parameters:
        cluster_labels_grid (list): list of shape (map_size, map_size) where each entry is the cluster to which the node in the sam eposition belongs to.
        map_size (int): Length of the edges of squared SOM. Map size is actually (map_size*map_size).
    Returns:
        clustered_genes_ids (dictionary)_ dictionary with every cluster and its corresponding gene IDs.
    """
    clustered_genes_ids={}
    for y in range(map_size):
        for x in range(map_size):
            try:
                if cluster_labels_grid[y][x] in clustered_genes_ids:
                    clustered_genes_ids[cluster_labels_grid[y][x]].extend(geneid_grid[(x, y)])
                else:
                    clustered_genes_ids[cluster_labels_grid[y][x]] = list(geneid_grid[(x, y)])
            except KeyError:
                continue
    return clustered_genes_ids
    

#%% 11. Evidence Accumulation KMeans

def evidence_accumulation(data, k, nensemble, linkage):
    """
    Performs Evidence Accumulation KMeans, and shows a plot of its clustering result,
    along with a barplot showing each cluster's individual silhouette score.
    Parameters:
        data (array): metagene_map
        k (int): Number of final clusters. If k=0, then the lifetime criterion is used.
        nensemble (int): Number of ensemble runs.
        linkage (str): Type of linkage algorithm. Options 'single', 'average', 'weighted', and 'complete'.
    """
    EAC=consensus_kmeans(data=data, k=k, nensemble=nensemble, linkage=linkage)
    matrix = np.zeros((40, 40), dtype=int)
    for key, indices in EAC[0].items():
        for index in indices:
            row = index // 40
            col = index % 40
            matrix[row, col] = key

    print("Silhouette Score: {}".format(silhouette_score(data, matrix.flatten())))
    print("Davies-Bouldin: {}".format(davies_bouldin_score(data, matrix.flatten())))
    print("Calinski-Harabasz: {}".format(calinski_harabasz_score(data, matrix.flatten())))
    custom_colormap=['#9e0142','#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#d9ef8b', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2', '#ffffff', '#878787', '#1a1a1a', '#c51b7d', '#b2abd2', '#4d9221', '#35978f', '#313695', '#8c510a']
    norm = plt.Normalize(0, len(custom_colormap) - 1)
    cmap = colors.ListedColormap(custom_colormap)
    plt.matshow((matrix), cmap=cmap, origin='lower')
    plt.title('KMeans Clustering of Metagenes')
    plt.colorbar(label='Cluster Label').set_ticks(np.arange(0,19,1))
    plt.show()

    clusters_of_interest = range(0,len(EAC[0]))
    mask = [label in clusters_of_interest for label in matrix.flatten()]
    X_selected = data[mask]
    labels_selected = np.array([label for label in matrix.flatten() if label in clusters_of_interest])
    silhouette_avg = silhouette_score(X_selected, labels_selected)
    silhouette_vals = silhouette_samples(X_selected, labels_selected)
    cluster_silhouette_vals = {}

    for k in clusters_of_interest:
        cluster_mask = (labels_selected == k)
        cluster_vals = silhouette_vals[cluster_mask]
        cluster_size = cluster_mask.sum()
        if cluster_size > 1: 
            cluster_silhouette_vals[k] = cluster_vals
        else:
            cluster_silhouette_vals[k] = np.array([]) 

    average_silhouette_scores = {k: (np.mean(v) if len(v) > 0 else np.nan) for k, v in cluster_silhouette_vals.items()}
    print(f"Average Silhouette Score within Cluster: {average_silhouette_scores}")
    print(f"Total Average: {silhouette_avg}")
    keys=[]
    items=[]
    for key in (average_silhouette_scores):
        keys.append(key)
        items.append(average_silhouette_scores[key])
    plt.bar(keys,items)
    plt.axhline(0.25, color='red', linestyle='--')
    plt.xticks(range(0,len(EAC[0]),1))
    plt.show
    return matrix

#%% 12. DrawApp

def Desenho(background_image_matrix, map_size, root = None, filename=''):
    """
    Opens window for cluster drawing with a user selected background. Saves the cluster as .npy file.
    Parameters:
        filename (str): Name of the file with the cluster mask.
        background (array): Must be an array with size (map_size, map_size). Can be either one of the SOM's or KMeans or EAKMeans.
    """
    if root is None:
        root=tk.Tk()
    app = DrawApp.DrawApp(root, map_size, mask_name=filename, background_image_matrix=background_image_matrix)
    root.mainloop()

def mycluster(input_file, main_map, genesymbol_grid, geneid_grid):
    """
    Retrieves the elements of the cluster previously defined.
    Parameters:
        inut_file (str): Name of the file with the mask previously drawn (.npy).
    Returns:
        mycluster_names (list): List of all the gene symbols assigned to the elements of the grid within the cluster.
        mycluster_ids (list): List of all the gene Ensembl IDs assigned to the elements of the grid within the cluster.
    """
    mask=np.load(input_file)
    mycluster_names=[]
    mycluster_ids=[]
    for y in range(main_map.map_size):
        for x in range(main_map.map_size):
            if mask[y,x]==1:
                try:
                    mycluster_names.extend(genesymbol_grid[(x, y)])
                    mycluster_ids.extend(geneid_grid[(x, y)])
                except KeyError as e:
                    print(f"Entry {e} is empty")
    return mycluster_names, mycluster_ids