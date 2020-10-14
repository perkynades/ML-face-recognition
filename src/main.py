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

# PCA for densionality reduction - reconstruct images using a subset of features
pca = PCA(n_components=8*8)
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)

x_reduced = pca.fit_transform(x_test)
x_recovered = pca.inverse_transform(x_reduced)

# Use autoencoder - reconstruct using a compressed representation (code)
# Call neural network API: sequential model is a linear stack of layers
autoencoder = Sequential()
autoencoder.add(Dense(units=1024, activation='relu', input_dim=64*64, name='encoder_layer1'))
autoencoder.add(Dense(units=512, activation='relu', input_dim=1024, name='encoder_layer2'))
autoencoder.add(Dense(units=64, activation='relu', input_dim=512, name='encoder_layer3'))

autoencoder.add(Dense(units=512, activation='relu', name='decoder_layer1'))
autoencoder.add(Dense(units=1024, activation='relu', name='decoder_layer2'))
autoencoder.add(Dense(units=64*64, activation='sigmoid', name='decoder_layer3'))

autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

autoencoder.summary()

history = autoencoder.fit(x = x_train, y = x_train, epochs=10, batch_size=32, shuffle=True, validation_data=(x_train, x_train), verbose=1)

# Plot accuracy and loss
# Test trained autoencoder
encoder = Model(inputs = autoencoder.input, outputs = autoencoder.layers[2].output)
encoded_imgs = encoder.predict(x_test)

print(encoded_imgs.shape)

# Retrieve the last layer of the autoencoder model

encoded_input = Input(shape=(64,))
decoder_layer = autoencoder.layers[3](encoded_input)
decoder_layer = autoencoder.layers[4](decoder_layer)
decoder_layer = autoencoder.layers[5](decoder_layer)

decoder = Model(inputs = encoded_input, outputs = decoder_layer)

decoded_imgs = decoder.predict(encoded_imgs)

print(encoded_imgs.shape)

predicted_imgs = autoencoder.predict(x_test, verbose=1)

n = 10  
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(64, 64))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display encoded images
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(encoded_imgs[i].reshape(8, 8))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i].reshape(64, 64))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
