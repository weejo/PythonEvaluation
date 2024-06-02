import json

import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.neighbors import NearestNeighbors

# Specify the path to your JSON file
file_path = './mnist.json'

# Open the file and load the JSON data
with open(file_path, 'r') as file:
    data = json.load(file)

X = pandas.DataFrame(data)

neighbors = NearestNeighbors(n_neighbors=10)  # You might need to adjust this based on your resources
neighbors.fit(X)
distances, indices = neighbors.kneighbors(X)

# Compute the average distance to the nearest points for each point
avg_distances = np.mean(distances, axis=1)
sorted_distances = np.sort(avg_distances)

plt.plot(sorted_distances)
plt.title('Elbow Plot for Optimal K in KNN on MNIST')
plt.xlabel('Sample index')
plt.ylabel('Average Distance to Nearest Neighbors')
plt.grid(True)
plt.show()
