import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression


import logistic_regression as lr

data_2 = pd.read_csv('data_2.csv')

data_2_train = data_2[data_2["split"] == "train"]
data_2_test = data_2[data_2["split"] == "test"]
# Partition data into independent (feature) and depended (target) variables

X_train = data_2[['x0', 'x1']]
y_train = data_2['y']

X_test = data_2_test[['x0', 'x1']]
y_test = data_2_test['y']

# Create and train model.
model_2 = lr.LogisticRegression(iterations=100,input_dimension=6) # <-- Should work with default constructor
model_2.fit(X_train, y_train)

# Calculate accuracy and cross entropy for (insample) predictions
y_pred_train = model_2.predict(X_train)
y_pred_test = model_2.predict(X_test)

print(f'Train Accuracy: {lr.binary_accuracy(y_true=y_train, y_pred=y_pred_train, threshold=0.5) :.3f}')
print(f'Train Cross Entropy: {lr.binary_cross_entropy(y_true=y_train, y_pred=y_pred_train) :.3f}')

print(f'Test Accuracy: {lr.binary_accuracy(y_true=y_test, y_pred=y_pred_test, threshold=0.5) :.3f}')
print(f'Test Cross Entropy: {lr.binary_cross_entropy(y_true=y_test, y_pred=y_pred_test) :.3f}')


# Rasterize the model's predictions over a grid
xx0, xx1 = np.meshgrid(np.linspace(-0.1, 1.1, 100), np.linspace(-0.1, 1.1, 100))
yy = model_2.predict(np.stack([xx0, xx1], axis=-1).reshape(-1, 2)).reshape(xx0.shape)

# Plot prediction countours along with datapoints
plt.switch_backend("TkAgg")
_, ax = plt.subplots(figsize=(5, 8), dpi=100)
levels = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
contours = ax.contourf(xx0, xx1, yy, levels=levels, alpha=0.4, cmap='RdBu_r', vmin=0, vmax=1)
legends = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in contours.collections]
labels = [f'{a :.2f} - {b :.2f}' for a,b in zip(levels, levels[1:])]
sns.scatterplot(x='x0', y='x1', hue='y', ax=ax, data=data_2)
ax.legend(legends, labels, bbox_to_anchor=(1,1));

plt.show()