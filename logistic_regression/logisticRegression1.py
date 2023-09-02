import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression


import logistic_regression as lr

data_1 = pd.read_csv('data_1.csv')
#data_1 = datasets.load_breast_cancer()
# Partition data into independent (feature) and depended (target) variables
print(data_1)
X = data_1[['x0', 'x1']]
y = data_1['y']

# Standardizing
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

X = (X - X_mean) / X_std


# Create and train model.
model_1 = lr.LogisticRegression() # <-- Should work with default constructor
model_1.fit(X, y)

# Calculate accuracy and cross entropy for (insample) predictions
y_pred = model_1.predict(X)
print(f'Accuracy: {lr.binary_accuracy(y_true=y, y_pred=y_pred, threshold=0.5) :.3f}')
print(f'Cross Entropy: {lr.binary_cross_entropy(y_true=y, y_pred=y_pred) :.3f}')


# Rasterize the model's predictions over a grid
xx0, xx1 = np.meshgrid(np.linspace(-0.1, 1.1, 100), np.linspace(-0.1, 1.1, 100))
yy = model_1.predict(np.stack([xx0, xx1], axis=-1).reshape(-1, 2)).reshape(xx0.shape)
#print("xx: ",xx)
# Plot prediction countours along with datapoints
plt.switch_backend("TkAgg")
_, ax = plt.subplots(figsize=(5, 8), dpi=100)
levels = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
contours = ax.contourf(xx0, xx1, yy, levels=levels, alpha=0.4, cmap='RdBu_r', vmin=0, vmax=1)
legends = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in contours.collections]
labels = [f'{a :.2f} - {b :.2f}' for a,b in zip(levels, levels[1:])]
sns.scatterplot(x='x0', y='x1', hue='y', ax=ax, data=data_1)
ax.legend(legends, labels, bbox_to_anchor=(1,1));

plt.show()