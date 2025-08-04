# Step 2: PCA (Principal Component Analysis)
# 2.1 Apply PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Keep a high number of components initially (e.g., all features)
pca = PCA()
X_pca = pca.fit_transform(X_scaled_df)

# Get explained variance ratio
explained_variance = pca.explained_variance_ratio_

# Cumulative variance
cumulative_variance = np.cumsum(explained_variance)

# Plot cumulative variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.grid()
plt.show()


# 2.2 Choosing Number of Components
# Keeping only components that explain 95% of variance
pca_95 = PCA(n_components=0.95)
X_pca_95 = pca_95.fit_transform(X_scaled_df)

print(f"Original shape: {X_scaled_df.shape}")
print(f"Reduced shape (95% variance): {X_pca_95.shape}")

# 2.3 Visualize 2D PCA
# 2D PCA for visualization
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled_df)

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='coolwarm', alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D PCA Projection of Heart Disease Data')
plt.colorbar(label='Target (0 = No Disease, 1 = Disease)')
plt.grid()
plt.show()