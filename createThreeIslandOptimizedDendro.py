import pandas as pd
import requests
from community import community_louvain
from sklearn.datasets import load_iris
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import numpy as np


basepath = "http://85.215.235.161:8080/getAllSolutions?levelId="

result = requests.get(basepath + "2")

percentage = 49.54

json = result.json()
solution = json[2]["solutionMatrix"]

iris = load_iris()
labels_true = iris.target

data = pd.DataFrame(solution)
df = data.iloc[:150, :150]

max_val = df.to_numpy().max()

# from 0 to 100
# Get current max
threshold = (max_val / 100) * percentage

# get copy of dataframe - default is deep copy
datacopy = df.copy()

# replace everything below threshold with 0, everything above or equal with 1
datacopy[datacopy < threshold] = 0
datacopy[datacopy >= threshold] = 1

# create networkx graph
G = nx.from_pandas_adjacency(datacopy)

# Calculate the transitive closure
TC = nx.transitive_closure(G)

# Getting the adjacency matrix of the graph
X = nx.to_numpy_array(TC)

# Check if the data has only one feature otherwise stuff breaks.
if X.shape[1] == 1:
    X = X.reshape(-1, 1)

# Extract communities via louvain
partition = community_louvain.best_partition(TC)

# Create a distance matrix for the linkage function
nodes = list(TC.nodes())
matrix_size = len(nodes)
distance_matrix = np.zeros((matrix_size, matrix_size))

for i in range(matrix_size):
    for j in range(matrix_size):
            if partition[nodes[i]] != partition[nodes[j]]:
                distance_matrix[i, j] = 1  # Different community

# Convert the distance matrix to condensed form
condensed_matrix = pdist(distance_matrix)

# Compute the linkage
linked = linkage(condensed_matrix, 'single')

# Create the dendrogram
dendrogram(linked, orientation='top', no_labels=True)
plt.title(f'Dendrogram for Three Islands Optimized at {percentage}%')
plt.xlabel('Data points')
plt.ylabel('Distance between clusters')
plt.ylim(0, 15)
plt.savefig('C:/Users/SirMopf/Desktop/plots/threeislandsopt/Dendrogram at ' + str(percentage) +'.png')
plt.show()

# Change width
plt.figure(figsize=(5, 24))

# Create the dendrogram
dendrogram(linked, orientation='right', leaf_font_size=10)
plt.title(f'Dendrogram for Three Islands Optimized at {percentage}%')
plt.ylabel('Data points')
plt.xlabel('Distance between clusters')
plt.xlim(0, 15)
plt.savefig('C:/Users/SirMopf/Desktop/plots/threeislandsopt/Dendrogram at ' + str(percentage) +' Full.png')
plt.show()