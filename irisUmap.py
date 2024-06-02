import numpy as np
import umap
import json

from sklearn import datasets

np.random.seed(5)

iris = datasets.load_iris()
X = iris.data
Y = iris.target
umapped_Iris = umap.UMAP().fit_transform(X, Y)

scaled = umapped_Iris * 10

print(scaled)

scaled = np.trunc(scaled)

formatted = [{"x": entry[0], "y": entry[1], "id":idx} for idx, entry in enumerate(scaled)]

encodedScaled = json.dumps(str(formatted))

print(encodedScaled)
