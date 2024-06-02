import json

import matplotlib.pyplot as plt
import pandas
import umap
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score, silhouette_score, adjusted_rand_score, fowlkes_mallows_score


# Specify the path to your JSON file
file_path = './mnist.json'

# Open the file and load the JSON data
with open(file_path, 'r') as file:
    data = json.load(file)

X = pandas.DataFrame(data)

# Load the Iris dataset
digits = datasets.load_digits(n_class=2, return_X_y=False, as_frame=False)

labels_true = digits.target


# Perform KMeans clustering
kmeans = KMeans(n_clusters=2)
y_pred = kmeans.fit_predict(X)

# Calculate clustering metrics
v_measure = v_measure_score(labels_true, y_pred)
silhouette = silhouette_score(X, y_pred)
rand_index = adjusted_rand_score(labels_true, y_pred)
fowlkes_mallows = fowlkes_mallows_score(labels_true, y_pred)

# Print metrics
print(f"V-Measure: {v_measure:.4f}")
print(f"Silhouette Index: {silhouette:.4f}")
print(f"Adjusted Rand Index: {rand_index:.4f}")
print(f"Fowlkes-Mallows Score: {fowlkes_mallows:.4f}")

# Plotting the results of KMeans

scatter = plt.scatter(X.x, X.y, c=y_pred, cmap='viridis', s=50, alpha=0.7)
plt.title('KMeans Clustering on UMAP-transformed MNIST Data')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.colorbar(scatter, label='Cluster ID', ticks=[0, 1])
plt.show()
