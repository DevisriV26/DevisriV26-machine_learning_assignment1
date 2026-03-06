import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

print("First 5 Rows:")
print(data.head())

print("\nColumns:")
print(data.columns)

# Selecting useful features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# -----------------------------
# Algorithm 1 : K-Means Clustering
# -----------------------------
kmeans = KMeans(n_clusters=5, random_state=0)

data['Cluster'] = kmeans.fit_predict(X)

plt.figure()

plt.scatter(X['Annual Income (k$)'],
            X['Spending Score (1-100)'],
            c=data['Cluster'])

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("K-Means Customer Clusters")

plt.show()


# -----------------------------
# Algorithm 2 : Hierarchical Clustering
# -----------------------------
plt.figure()

linked = linkage(X, method='ward')

dendrogram(linked)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")

plt.show()
