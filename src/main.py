import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')
print("Warnings ignored!")

# Fetch data and have a look
data = np.load("src/olivetti_faces.npy")
target = np.load("src/olivetti_faces_target.npy")

def show_40_distinct_people(images, unique_ids):
    fig, axarr = plt.subplots(nrows=4, ncols=10, figsize=(18, 9))
    axarr = axarr.flatten()

    for unique_id in unique_ids:
        image_index = unique_id*10
        axarr[unique_id].imshow(images[image_index], cmap='gray')
        axarr[unique_id].set_xticks([])
        axarr[unique_id].set_yticks([])
        axarr[unique_id].set_title("face id:{}".format(unique_id))
    plt.suptitle("There are 40 distinct people in the dataset")

    plt.show()

def show_10_faces_of_n_subject(images, subject_ids):
    cols = 10
    rows = (len(subject_ids)*10)/cols
    rows = int(rows)

    fig, axarr = plt.subplots(nrows=rows, ncols=cols, figsize=(18, 9))

    for i, subject_id in enumerate(subject_ids):
        for j in range(cols):
            image_index = subject_id*10 + j
            axarr[i, j].imshow(images[image_index], cmap="gray")
            axarr[i, j].set_xticks([])
            axarr[i, j].set_yticks([])
            axarr[i, j].set_title("face id:{}".format(subject_id))
    
    plt.show()

# divide data into training, validation and testing (shuffle)
X = data.reshape((data.shape[0], data.shape[1]*data.shape[2]))
print("X shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.3, stratify=target, random_state=0)
print("X_train shape:", X_train.shape)
print("y_train shape:{}".format(y_train.shape))

y_frame=pd.DataFrame()
y_frame['subject ids']=y_train
y_frame.groupby(['subject ids']).size().plot.bar(figsize=(15,8),title="Number of Samples for Each Classes")

# use PCA for diensionality reduction - reconstruct images using a subset of features
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

number_of_people = 10
index_range = number_of_people*10
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
scatter = ax.scatter(X_pca[:index_range, 0],
                    X_pca[:index_range, 1],
                    c = target[:index_range],
                    s = 10,
                    cmap= plt.get_cmap('jet', number_of_people))
ax.set_xlabel("First principle component")
ax.set_ylabel("Second principle component")
ax.set_title("PCA projection of {} peope".format(number_of_people))
fig.colorbar(scatter)
plt.show()

# use autoencoder - reconstruct using a compressed representation (code)

# unsupervised algorithm

# supervised algorithm

# confusion matrix

#cross-validation  - SGD, BGD and MBGD