import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import k_means as km # <-- Your implementation

sns.set_style('darkgrid')

data1 = pd.read_csv("data_1.csv")


plt.figure(figsize=(5, 5))
sns.scatterplot(x='x0', y='x1', data=data1)
plt.xlim(0, 1); plt.ylim(0, 1);
print(data1.describe().T)

print(data1["x1"][0])
X = data1[["x0", "x1"]]
kmeans = km.KMeans()
kmeans.fit(X)

kmeans.get_centroids()



plt.show()


#model = kmeans.fit(data1)