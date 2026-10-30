import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

def run_precision_clustering(embeddings_list):
    if len(embeddings_list) < 2: return np.array([0] * len(embeddings_list))
    data = normalize(np.array(embeddings_list))
    clusterer = AgglomerativeClustering(
        n_clusters=None, metric='cosine', linkage='average', distance_threshold=0.35 
    )
    return clusterer.fit_predict(data)

def run_superfolder_merging(cluster_centroids):
    if len(cluster_centroids) < 2: return np.array([0] * len(cluster_centroids))
    data = normalize(cluster_centroids)
    clusterer = AgglomerativeClustering(
        n_clusters=None, metric='cosine', linkage='complete', distance_threshold=0.65 
    )
    return clusterer.fit_predict(data)