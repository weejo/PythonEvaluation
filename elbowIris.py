import json

import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors

# Specify the path to your JSON file
file_path = './iris.json'

# Open the file and load the JSON data
with open(file_path, 'r') as file:
    data = json.load(file)

X = pandas.DataFrame(data)

# Load the Iris dataset
iris = datasets.load_iris()

y_true = iris.target

neighbors = NearestNeighbors(n_neighbors=10)  # Adjust the range of K as needed
neighbors.fit(X)
distances, indices = neighbors.kneighbors(X)

# Compute the average distance to the nearest points for each point
avg_distances = np.mean(distances, axis=1)
sorted_distances = np.sort(avg_distances)

plt.plot(sorted_distances)
plt.title('Elbow Plot for Optimal K')
plt.xlabel('Sample index')
plt.ylabel('Average Distance to Nearest Neighbors')
plt.grid(True)
plt.show()
