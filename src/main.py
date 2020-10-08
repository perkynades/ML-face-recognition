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

show_40_distinct_people(data, np.unique(target))
# divide data into training, validation and testing (shuffle)

# use PCA for diensionality reduction - reconstruct images using a subset of features

# use autoencoder - reconstruct using a compressed representation (code)

# unsupervised algorithm

# supervised algorithm

# confusion matrix

#cross-validation  - SGD, BGD and MBGD