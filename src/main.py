import warnings
warnings.filterwarnings('ignore')
print("Warnings ignored!!")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets.olivetti_faces import fetch_olivetti_faces
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Activation, Dropout
from keras.utils import plot_model
from keras import regularizers
from keras.losses import mse, binary_crossentropy
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score

faces = fetch_olivetti_faces()
x, y = faces['data'], faces['target']

# Divide data into training, validation and testing (shuffle)
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)