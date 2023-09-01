
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import k_means as km


sns.set_style('darkgrid')

# Load data
data_2 = pd.read_csv('data_2.csv')

X = data_2[["x0", "x1"]]

# Make to Numpy array
X = np.asarray(X)

# Normalize data
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

X = (X-X_mean)/X_std


# Initialize and fit model
model_2 = km.KMeans(8,10)  # <-- Feel free to add hyperparameters
model_2.fit(X)

# Compute Silhouette Score
z = model_2.predict(X)

#print(f'Distortion: {km.euclidean_distortion(X, z) :.3f}')
print(f'Silhouette Score: {km.euclidean_silhouette(X, z) :.3f}')

# Inverse normalize
X = X * X_std + X_mean

# Inverse normalize centroids
C = model_2.get_centroids() * X_std + X_mean



# Plot cluster assignments
K = len(C)

_, ax = plt.subplots(figsize=(5, 5), dpi=100)
sns.scatterplot(x=X[:,0], y=X[:,1], hue=z, hue_order=range(K), palette='tab10', ax=ax);
sns.scatterplot(x=C[:,0], y=C[:,1], hue=range(K), palette='tab10', marker='*', s=250, edgecolor='black', ax=ax)
ax.legend().remove()

plt.show()