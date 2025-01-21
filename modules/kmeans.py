from sklearn.cluster import KMeans
import logging
import numpy as np

def apply_kmeans(data, n_clusters=3):
    logging.info(f"Dimensiuni date originale pentru K-Means: {data.shape}")
    logging.info(f"Tipurile de date înainte de K-Means:\n{data.dtypes}")

    # Filtrăm doar coloanele numerice
    numeric_data = data.select_dtypes(include=[np.number])
    logging.info(f"Dimensiuni date numerice pentru K-Means: {numeric_data.shape}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(numeric_data)

    # Adăugăm clusterele în DataFrame-ul original
    data['Cluster'] = clusters
    return data, kmeans
