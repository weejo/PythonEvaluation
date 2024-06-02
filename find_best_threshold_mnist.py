import requests
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from community import community_louvain
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn import datasets
from sklearn.metrics import v_measure_score
from sklearn.metrics import fowlkes_mallows_score


basepath = "http://85.215.235.161:8080/getAllSolutions?levelId="

result = requests.get(basepath + "4")

json = result.json()
solution = json[0]["solutionMatrix"]

digits = datasets.load_digits(n_class=2, return_X_y=False, as_frame=False)
labels_true = digits.target

df = pd.DataFrame(solution)

max_val = df.to_numpy().max()

rand_scores = []
SI_scores = []
v_measure_scores= []
fowlkes_mallows_scores= []


# from 0 to 100
for percentage in range(100):
    # Get current max
    threshold = (max_val / 100) * percentage

    # get copy of dataframe - default is deep copy
    datacopy = df.copy()

    # replace everything below threshold with 0, everything above or equal with 1
    datacopy[datacopy < threshold] = 0
    datacopy[datacopy >= threshold] = 1

    # create networkx graph
    G = nx.from_pandas_adjacency(datacopy)

    #Calculate the transitive closure
    TC = nx.transitive_closure(G)

    # Getting the adjacency matrix of the graph
    X = nx.to_numpy_array(TC)

    # Check if the data has only one feature otherwise stuff breaks.
    if X.shape[1] == 1:
        X = X.reshape(-1, 1)

    # Extract communities via louvain
    partition = community_louvain.best_partition(TC)
    labels_pred = [partition[i] for i in range(len(G.nodes))]

    # Silhouette score
    SI_scores.append(silhouette_score(X, labels_true))

    # Adjusted Rand score
    rand_scores.append(adjusted_rand_score(labels_true, labels_pred))

    # V-measure score
    v_measure_scores.append(v_measure_score(labels_true, labels_pred))

    # Calinski score
    fowlkes_mallows_scores.append(fowlkes_mallows_score(labels_true, labels_pred))


plt.rcParams.update({'font.size': 12})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

sns.lineplot(x=range(100), y=rand_scores, marker='o', color='green', ax=ax1)
ax1.set_title('Adjusted Rand Index')
ax1.set_xlabel('Threshold in %')
ax1.set_ylabel('Adjusted Rand Score')
ax1.grid(True)
ax1.set_xticks(np.arange(0, 101, 10.0))

sns.lineplot(x=range(100), y=SI_scores, marker='o', color='green', ax=ax2)
ax2.set_title('Silhouette Index')
ax2.set_xlabel('Threshold in %')
ax2.set_ylabel('Silhouette Score')
ax2.grid(True)
ax2.set_xticks(np.arange(0, 101, 10.0))

plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

sns.lineplot(x=range(100), y=v_measure_scores, marker='o', color='green', ax=ax1)
ax1.set_title('V-Measure')
ax1.set_xlabel('Threshold in %')
ax1.set_ylabel('V-Measure Score')
ax1.grid(True)
ax1.set_xticks(np.arange(0, 101, 10.0))

sns.lineplot(x=range(100), y=fowlkes_mallows_scores, marker='o', color='green', ax=ax2)
ax2.set_title('Fowlkes–Mallows Index')
ax2.set_xlabel('Threshold in %')
ax2.set_ylabel('FMI Score')
ax2.grid(True)
ax2.set_xticks(np.arange(0, 101, 10.0))

plt.tight_layout()
plt.show()

