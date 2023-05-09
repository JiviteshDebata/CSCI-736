from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import random

# r^2 = x^2 + y^2 + z^2 + k^2
X = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [1, 0, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 1],
    [1, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [0, 1, 1, 1],
])

random.shuffle(X)


# Create an instance of the PCA class with 2 components
pca = PCA(n_components=2)

# Fit the PCA model to the data
pca.fit(X)

# Transform the data to the new coordinate system defined by the first 2 principal components
X_pca = pca.transform(X)

# Print the original and transformed data
print("Original data:")
print(X)
print("Transformed data:")
print(X_pca)


# Plot the transformed data
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Transformed Data')
plt.show()
