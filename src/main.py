import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.olivetti_faces import fetch_olivetti_faces

# Fetch data and have a look
faces = fetch_olivetti_faces()
x, y = faces['data'], faces['target']
print(f'Data shape: {x.shape}')
print(f'Label shape: {y.shape}')

# Create a grid of 3x3 images
for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x[i].reshape(64, 64))

plt.show()

# Divide data into training, validation and testing (shuffle)

# Use PCA for diensinality reduction - reconstruct images using a subset of features

# Use autoencode - reconstruct using a compressed representation (code)

# Unsupervised algorithm

# Supervised algorithm

# Confusion matrix

# Cross - validation - SGD, BGD and MBGD