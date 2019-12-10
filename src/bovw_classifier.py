# %% Import libraries
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import os
sys.path.append('src')
from utilities import load_training_images, load_testing_images
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import random
from sklearn.preprocessing import MinMaxScaler
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
        feature_vectors.append(feature_vector[::4]) # Sampling every 4 px
    return np.asarray(feature_vectors)

# %% Load images
(train_images, train_image_classes, class_names) = load_training_images()
test_images = load_testing_images()

# For testing only
sample_indices = np.random.randint(0, len(train_images), 50)
train_images = np.array(train_images)[sample_indices]
train_image_classes = np.array(train_image_classes)[sample_indices]
test_images = random.sample(test_images, 10)

# %% Training image sample
fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(9,12))
for i in range(3):
    for j in range(4):
        ax[i][j].set_title(class_names[train_image_classes[(3 * i) + j] - 1])
        ax[i][j].imshow(train_images[(3 * i) + j], cmap='gray')
        ax[i][j].set_xticks([],[])
        ax[i][j].set_yticks([],[])

# %% Get feature vectors and normalize
train_feature_vectors = []
for img in train_images:
    features = extract_features(img)
    for f in features:
         train_feature_vectors.append(f)
scaler = MinMaxScaler()
train_feature_vectors = scaler.fit_transform(np.asarray(train_feature_vectors))

# %% Clustering
K = 100 # For testing
kmeans = KMeans(K).fit(train_feature_vectors)
codewords = kmeans.cluster_centers_
labels = kmeans.labels_

# %% Sample codewords
fig, ax = plt.subplots(nrows=1, ncols=20, figsize=(18,3))
for i in range(20):
    ax[i].imshow(np.reshape(codewords[i], (4,4)), cmap='gray')
    ax[i].set_xticks([],[])
    ax[i].set_yticks([],[])

# %% Generate image histogram
def build_img_histogram(img, kmeans):
    scaler = MinMaxScaler()
    features = scaler.fit_transform(extract_features(train_images[i]))
    cluster_predict = kmeans.predict(features)
    histogram = np.histogram(cluster_predict, bins=range(len(kmeans.cluster_centers_) + 1))[0]
    return histogram

# %% Generate historgrams of train and test images
train_img_histograms = np.zeros((len(train_images), K))
for i in range(len(train_images)):
    train_img_histograms[i] = build_img_histogram(train_images[i], kmeans)

test_img_histograms = np.zeros((len(test_images), K))
for i in range(len(test_images)):
    test_img_histograms[i] = build_img_histogram(test_images[i], kmeans)

# %% Sample histogram
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(18,3))
hist_1 = train_img_histograms[np.where(train_image_classes == 13)[0],:]
for i in range(min(4,len(hist_1))):
    ax[i].set_title(class_names[train_image_classes[12] - 1])
    ax[i].bar(range(0, K), hist_1[i])

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(18,3))
hist_2 = train_img_histograms[np.where(train_image_classes == 6)[0],:]
for i in range(min(4,len(hist_2))):
    ax[i].set_title(class_names[train_image_classes[5] - 1])
    ax[i].bar(range(0, K), hist_2[i])

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(18,3))
hist_3= train_img_histograms[np.where(train_image_classes == 15)[0],:]
for i in range(min(4,len(hist_3))):
    ax[i].set_title(class_names[train_image_classes[14] - 1])
    ax[i].bar(range(0, K), hist_3[i])

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(18,3))
hist_4= train_img_histograms[np.where(train_image_classes == 4)[0],:]
for i in range(min(4,len(hist_4))):
    ax[i].set_title(class_names[train_image_classes[4] - 1])
    ax[i].bar(range(0, K), hist_4[i])

# %% Build classifiers
classifiers = SVC()
classifiers.fit(train_img_histograms, train_image_classes)
predicted = classifiers.predict(test_img_histograms)

# %% Classify test images
if os.path.exists('run2.txt'):
    os.remove('run2.txt')
f = open('run2.txt', 'x')
for i in range(len(test_img_histograms)):
    plt.figure()
    img_name = class_names[predicted[i] - 1]
    plt.title(img_name)
    plt.imshow(test_images[i], cmap='gray')
    f = open('run2.txt', 'a')
    f.write(f'{i}.jpg {img_name}' + '\n')