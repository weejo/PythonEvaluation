import json

import matplotlib.pyplot as plt
import pandas
from sklearn import datasets
from sklearn.cluster import KMeans
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

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3)
y_pred = kmeans.fit_predict(X)

# Calculate V-Measure
v_measure = v_measure_score(labels_true, y_pred)

# Calculate Silhouette Index
silhouette = silhouette_score(X, y_pred)

# Calculate Adjusted Rand Index
rand_index = adjusted_rand_score(labels_true, y_pred)

# Calculate Fowlkes-Mallows scores
fowlkes_mallows = fowlkes_mallows_score(labels_true, y_pred)

# Print the scores
print(f"V-Measure: {v_measure:.4f}")
print(f"Silhouette Index: {silhouette:.4f}")
print(f"Adjusted Rand Index: {rand_index:.4f}")
print(f"Fowlkes-Mallows Score: {fowlkes_mallows:.4f}")

plt.scatter(X.x, X.y, c=y_pred, cmap='viridis', s=50)
plt.title('KMeans Clustering on UMAP-reduced Iris dataset')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.colorbar(label="Cluster ID", ticks=[0, 1, 2])
plt.show()
