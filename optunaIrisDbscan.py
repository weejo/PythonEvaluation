import json

import numpy as np
import optuna
import pandas
import umap
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.metrics import v_measure_score, silhouette_score, adjusted_rand_score, fowlkes_mallows_score

# Specify the path to your JSON file
file_path = './mnist.json'

# Open the file and load the JSON data
with open(file_path, 'r') as file:
    data = json.load(file)

X = pandas.DataFrame(data)

digits = datasets.load_digits(n_class=2, return_X_y=False, as_frame=False)

labels_true = digits.target


def objective(trial):
    # Suggest a value for eps within a specified range
    eps = trial.suggest_uniform('eps', 20, 100)

    # DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(X)

    # Check the number of clusters formed
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Avoid invalid clustering results
    if n_clusters < 2:
        return -1  # Invalid result due to having fewer than 2 clusters

    # Calculate metrics
    silhouette = silhouette_score(X, labels)
    rand_index = adjusted_rand_score(labels_true, labels)
    v_measure = v_measure_score(labels_true, labels)
    fmi = fowlkes_mallows_score(labels_true, labels)

    # Composite objective: average of normalized scores (Optionally, weights can be added)
    return (silhouette + rand_index + v_measure + fmi) / 4


# Create a study object and specify the optimization direction
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)  # Adjust the number of trials as needed

# Best parameters and score
print("Best parameters: ", study.best_params)
print("Best score: ", study.best_value)
