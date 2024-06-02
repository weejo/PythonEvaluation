import json

import matplotlib.pyplot as plt
import pandas
from sklearn import datasets
from sklearn.cluster import DBSCAN
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import v_measure_score, silhouette_score, adjusted_rand_score, fowlkes_mallows_score

# Specify the path to your JSON file
file_path = './iris.json'

# Open the file and load the JSON data
with open(file_path, 'r') as file:
    data = json.load(file)

X = pandas.DataFrame(data)

# Load the Iris dataset
iris = datasets.load_iris()

labels_true = iris.target

dbscan = DBSCAN(eps=11, min_samples=5)  # Adjust min_samples as needed
labels_pred = dbscan.fit_predict(X)


# Initialize the adjacency matrix with zeros
n_samples = len(data)
adj_matrix = np.zeros((n_samples, n_samples))

# Fill the adjacency matrix based on cluster labels
for i in range(n_samples):
    for j in range(n_samples):
        if labels_pred[i] == labels_pred[j]:
            adj_matrix[i, j] = 1


# Convert the distance matrix to condensed form
condensed_matrix = pdist(adj_matrix)

# Compute the linkage
mnist_res = linkage(condensed_matrix, 'single')

# Plotting the results of KMeans
plt.rcParams.update({'font.size': 12})


# Change width

# Create the dendrogram
dendrogram(mnist_res, orientation='top', no_labels=True)
plt.title(f'Dendrogram DBSCAN on Iris data')
plt.xlabel('Data points')
plt.ylabel('Similarity Distance')
plt.ylim(0, 15)
plt.show()
