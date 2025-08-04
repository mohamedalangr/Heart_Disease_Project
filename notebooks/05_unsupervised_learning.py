# Step 5: Unsupervised Learning – Clustering
# 5.1 K-Means Clustering
# Step 1: Use Elbow Method to Choose Optimal k
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Try k from 1 to 10
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X_selected)
    inertia.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means")
plt.grid()
plt.show()

# Step 2: Fit K-Means with Chosen k
# Choose k (let’s assume 2 clusters)
kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
kmeans_labels = kmeans.fit_predict(X_selected)

# Add to DataFrame (optional)
X_kmeans = X_selected.copy()
X_kmeans['Cluster'] = kmeans_labels

# Step 3: Visualize Clusters (with PCA)
from sklearn.decomposition import PCA

pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_selected)

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
plt.title("K-Means Clustering (2D PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid()
plt.show()


# 5.2 Hierarchical Clustering
# Step 1: Plot Dendrogram
from scipy.cluster.hierarchy import linkage, dendrogram

# Use Ward's method
linked = linkage(X_selected, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Euclidean Distance")
plt.show()


# Step 2: Fit Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=2)
agg_labels = agg.fit_predict(X_selected)

# Optional: Compare to target
from sklearn.metrics import adjusted_rand_score
print("Adjusted Rand Index (ARI) vs. true labels:", adjusted_rand_score(y, agg_labels))