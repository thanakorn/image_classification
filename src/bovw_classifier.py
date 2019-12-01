# %% Import libraries
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mimg
sys.path.append('src')
from utilities import load_training_images, load_testing_images
from sklearn.cluster import KMeans
import random
%matplotlib inline

# %% Extract features
PATCH_SIZE = (8,8)
def extract_patches(img, patch_size):
    x_size = patch_size[1]
    y_size = patch_size[0]
    patches = []
    for i in range(int(img.shape[0] / y_size)):
        for j in range(int(img.shape[1] / x_size)):
            patch = img[i * y_size:(i + 1) * y_size , j * x_size:(j + 1) * x_size]
            patches.append(patch)
    return patches

def extract_features(img):
    feature_vectors = []
    patches = extract_patches(img, PATCH_SIZE)
    for p in patches:
        feature_vector = p.flatten() 
        feature_vectors.append(feature_vector[::4]) # Downsampling every 4 pxl
    return np.asarray(feature_vectors)

# %% Load images
train_images = load_training_images()
test_images = load_testing_images()

# For testing only
train_images = random.sample(train_images, 50)
test_images = random.sample(test_images, 10)

print('Training image sample')
for i in np.random.randint(0, len(train_images), 10):
    plt.figure()
    plt.imshow(train_images[i], cmap='gray')

# %% Get feature vectors
train_data = []
for img in train_images:
    features = extract_features(img)
    for f in features:
         train_data.append(f)
train_data = np.asarray(train_data)

# %% Cluster descriptors
K = 30 # For testing
group_descriptors = KMeans(K).fit(train_data).cluster_centers_

# %% Sample image vocabularies
print('Sample vocabularies')
for v in group_descriptors[0:5,:]:
    plt.figure()
    plt.imshow(np.trunc(np.reshape(v, (4,4))).astype(int), cmap='gray')

# %%
