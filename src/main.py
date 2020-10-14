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

history = autoencoder.fit(x = x_train, y = x_train, epochs=200, batch_size=32, shuffle=True, validation_data=(x_train, x_train), verbose=1)

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
plt.show(block=False)

# Unsupervised algorithm
# Create KMC classifier
kmeans = KMeans(n_clusters=40, max_iter=10000, algorithm='auto', random_state=7, verbose=1)

# Train the model using the training sets
encoded_imgs = encoder.predict(x_train)
kmeans.fit(encoded_imgs)

# Predict the response for test dataset
encoded_imgs = encoder.predict(x_test)
y_pred = kmeans.predict(encoded_imgs)
print(y_pred[0,])
print(y_test[0,])

# Model accuracy, how often is the classifier correct?
print("Accuracy:", accuracy_score(y_test, y_pred))

cmap = ListedColormap(['lightgrey', 'silver', 'ghostwhite', 'lavender', 'wheat'])
plt.rcParams["figure.figsize"] = (8, 8)

# Confusion matrix
def plot_cm(ytest, ypred, title):
    cm = confusion_matrix(ytest, ypred)

    plt.matshow(cm, cmap=cmap)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(x=j, y=i, s=cm[i,j], va='center', ha='center')

    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show(block=False)

plot_cm(y_train, kmeans.predict(encoder.predict(x_train)), title='Train')
plot_cm(y_test, kmeans.predict(encoder.predict(x_test)), title='Test')

# Supervised algorithm
print(y.shape)
print(y_train[0])
print(y_train.shape)

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
print(y_train_cat[0])

model = Sequential()
model.add(Dense(56, activation='relu', input_shape=(8*8,), name='hidden_layer'))
model.add(Dense(40, activation='softmax', name='output_layer'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
plot_model(model, to_file='mnist_keras.png', show_shapes=True, show_layer_names=True)

enc_train = encoder.predict(x_train).reshape(320, 8*8)
enc_test = encoder.predict(x_test).reshape(80, 8*8)
print(enc_train.shape)
print(enc_test.shape)

history = model.fit(enc_train, y_train_cat, epochs=500, batch_size=32)
test_loss, test_accu = model.evaluate(enc_test, y_test_cat)
print(test_loss, test_accu)

# Confusion matrix
y_pred_train = model.predict(enc_train)
y_pred_train = np.argmax(y_pred_train, axis=-1)
plot_cm(y_train, y_pred_train, title='Train')

y_pred_test = model.predict(enc_test)
y_pred_test = np.argmax(y_pred_test, axis=-1)
print(y_pred_test.shape)
print(y_test.shape)

print(np.unique(y_pred_test))
print(np.unique(y_test))

print('Precision score:', precision_score(y_test, y_pred_test, average='macro'))
print('Recall score   :', recall_score(y_test, y_pred_test, average='macro'))

plot_cm(y_test[1:], y_pred_test[1:], title='Test')

plt.show()