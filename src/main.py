import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.olivetti_faces import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

# Fetch data and have a look
faces = fetch_olivetti_faces()
x, y = faces['data'], faces['target']
print(f'Data shape: {x.shape}')
print(f'Label shape: {y.shape}')

# Create a grid of 3x3 images
# for i in range(0, 9):
#     plt.subplot(330 + 1 + i)
#     plt.imshow(x[i].reshape(64, 64))

# plt.show()

# Divide data into training, validation and testing (shuffle)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
x_test, x_validate, y_test, y_validate = train_test_split(x_test, y_test, test_size=0.5)

print('x_test size:', x_test.size)
print('y_test size:', y_test.size)
# Use PCA for diensinality reduction - reconstruct images using a subset of features

# Use autoencode - reconstruct using a compressed representation (code)

# Unsupervised algorithm

# Supervised algorithm

# Confusion matrix

# Cross - validation - SGD, BGD and MBGD