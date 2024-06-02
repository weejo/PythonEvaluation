import numpy as np
import umap
import json

from sklearn import datasets

np.random.seed(5)

digits = datasets.load_digits(n_class=2, return_X_y=False, as_frame=False)
X = digits.data
Y = digits.target
umapped = umap.UMAP().fit_transform(X, Y)

scaled = umapped * 20

print(scaled)

scaled = np.trunc(scaled)

formatted = [{"x": entry[0], "y": entry[1], "id":idx} for idx, entry in enumerate(scaled)]

encodedScaled = json.dumps(str(formatted))

print(encodedScaled)
