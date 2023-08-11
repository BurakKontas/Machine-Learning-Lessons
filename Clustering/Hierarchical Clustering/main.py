# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.title("Dendrogram")
plt.xlabel("Observation Points")
plt.ylabel("Euclidean Distances")
plt.show()


# Optimal number of clusters is 5 due to graph
# Training the model
from sklearn.cluster import AgglomerativeClustering

clusters = 5
hc = AgglomerativeClustering(n_clusters=clusters, metric="euclidean", linkage="ward")
y_hc = hc.fit_predict(X)

# Visualising the clusters
colors = ["red","blue","green","cyan","magenta"]
for i in range(clusters):
    plt.scatter(X[y_hc == i, 0], X[y_hc == i, 1], s = 100, c = colors[i], label = f'Cluster {i+1}')
    
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()