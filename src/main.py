import numpy as np
import matplotlib.pyplot as plt

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

show_10_faces_of_n_subject(images=data, subject_ids=[0, 5, 21, 24, 36])
# divide data into training, validation and testing (shuffle)

# use PCA for diensionality reduction - reconstruct images using a subset of features

# use autoencoder - reconstruct using a compressed representation (code)

# unsupervised algorithm

# supervised algorithm

# confusion matrix

#cross-validation  - SGD, BGD and MBGD