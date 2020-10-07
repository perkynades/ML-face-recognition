import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.olivetti_faces import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

# Fetching faces data, and converting faces data from matrix to vector 
data = np.load("datasets/olivetti_faces.npy")
target = np.load("datasets/olivetti_faces.npy")

X = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
print("X shape:", X.shape)

# Divide data into training, validation and testing (shuffle)

# Use PCA for diensinality reduction - reconstruct images using a subset of features

# Use autoencode - reconstruct using a compressed representation (code)

# Unsupervised algorithm

# Supervised algorithm

# Confusion matrix

# Cross - validation - SGD, BGD and MBGD