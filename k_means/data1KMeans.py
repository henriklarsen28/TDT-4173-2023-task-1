import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import k_means as km # <-- Your implementation

sns.set_style('darkgrid')

data_1 = pd.read_csv('data_1.csv')

X = data_1[["x0", "x1"]]

# Make to Numpy array
X = np.asarray(X)

# Fit Model
model_1 = km.KMeans(2)  # <-- Feel free to add hyperparameters
model_1.fit(X)

# Predict Clusters
z = model_1.predict(X)

# Compute Silhouette Score
#print(f'Distortion: {km.euclidean_distortion(X, z) :.3f}')
print(f'Silhouette Score: {km.euclidean_silhouette(X, z) :.3f}')

# Get centroids
C = model_1.get_centroids()

# Plot cluster assignments
K = len(C)

_, ax = plt.subplots(figsize=(5, 5), dpi=100)
sns.scatterplot(x=X[:,0], y=X[:,1], hue=z, hue_order=range(K), palette='tab10', ax=ax);
sns.scatterplot(x=C[:,0], y=C[:,1], hue=range(K), palette='tab10', marker='*', s=250, edgecolor='black', ax=ax)
ax.legend().remove()

plt.show()

