import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import k_means as km
import sklearn.cluster as kmeans
from sklearn import preprocessing

sns.set_style('darkgrid')

scaler = StandardScaler()
data_2 = pd.read_csv('data_2.csv')


#print(data_2.describe().T)
# Fit Model
X = data_2[["x0", "x1"]]
X = scaler.fit_transform(X)

model_2 = km.KMeans(2)  # <-- Feel free to add hyperparameters
model_2.fit(X)


#model = kmeans.KMeans(n_clusters=2)
#model.fit(X)
#z = model.predict(X)

# Compute Silhouette Score
z = model_2.predict(X)

X = scaler.inverse_transform(X)

#print(f'Distortion: {km.euclidean_distortion(X, z) :.3f}')
print(f'Silhouette Score: {km.euclidean_silhouette(X, z) :.3f}')

# Plot cluster assignments
C = scaler.inverse_transform(model_2.get_centroids())
K = len(C)
_, ax = plt.subplots(figsize=(5, 5), dpi=100)
sns.scatterplot(x=X[:,0], y=X[:,1], hue=z, hue_order=range(K), palette='tab10', ax=ax);
sns.scatterplot(x=C[:,0], y=C[:,1], hue=range(K), palette='tab10', marker='*', s=250, edgecolor='black', ax=ax)
ax.legend().remove()

plt.show()