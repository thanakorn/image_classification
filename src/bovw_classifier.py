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
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC, LinearSVC
import random
from sklearn.feature_extraction import image
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split, KFold
from numpy import linalg as LA
from sklearn.metrics import accuracy_score
%matplotlib inline

# %% Extract features
PATCH_SIZE = (8,8)
MAX_PATCH = 256
SAMPLING_RATE = 4
# def extract_patches(img, patch_size):
#     x_size = patch_size[1]
#     y_size = patch_size[0]
#     patches = []
#     for i in range(int(img.shape[0] / y_size)):
#         for j in range(int(img.shape[1] / x_size)):
#             patch = img[i * y_size:(i + 1) * y_size , j * x_size:(j + 1) * x_size]
#             patches.append(patch)
#     return patches

def extract_features(img):
    patches = image.extract_patches_2d(img, PATCH_SIZE, max_patches=MAX_PATCH, random_state=42)
    feature_vectors = np.zeros((MAX_PATCH, (int)(PATCH_SIZE[0] * PATCH_SIZE[1] / SAMPLING_RATE)))
    for i in range(len(patches)):
        feature_vector = patches[i].flatten()[::SAMPLING_RATE] # Sampling every 4 px
        mean = feature_vector.mean()
        std = feature_vector.std()
        normed_feature_vector = (feature_vector - feature_vector.mean())
        if(std > 0): normed_feature_vector = normed_feature_vector / std
        feature_vectors[i] = normed_feature_vector
    return feature_vectors

# %% Load images
(all_train_images, all_train_image_labels, class_names) = load_training_images()
test_images = load_testing_images()

train_images = all_train_images
train_image_labels = all_train_image_labels

NUM_CLASS = len(class_names)
NUM_IMG_EACH_CLASS = 100

# %% Sample train images(Test Only)
# NUM_CLASS = 15
# NUM_IMG_EACH_CLASS = 5
# sample_indices = []
# for i in range(NUM_CLASS):
#     for j in range(NUM_IMG_EACH_CLASS):
#         sample_indices.append(i * 100 + j)

# train_images = np.array(train_images)[sample_indices]
# train_image_classes = np.array(train_image_classes)[sample_indices]
# test_images = random.sample(test_images, 10)

# %% Training image sample
fig, ax = plt.subplots(nrows = 15, ncols = 3, figsize=(6,30))
random_indices = np.random.randint(low = 0, high = NUM_IMG_EACH_CLASS, size = 3)
for i in range(NUM_CLASS):
    for j in range(len(random_indices)):
        ax[i][j].set_title(class_names[all_train_image_labels[(i * NUM_IMG_EACH_CLASS) + random_indices[j]] - 1])
        ax[i][j].imshow(train_images[(i * NUM_IMG_EACH_CLASS) + random_indices[j]], cmap='gray')
        ax[i][j].set_xticks([],[])
        ax[i][j].set_yticks([],[])

# %% Verified image sample
for i in np.random.randint(low=0, high=len(verified_images), size = 10):
    plt.figure()
    plt.title(class_names[verified_image_classes[i] - 1])
    plt.imshow(verified_images[i], cmap='gray')

# %% Test image sample
for i in np.random.randint(low=0, high=len(test_images), size = 10):
    plt.figure()
    plt.imshow(test_images[i], cmap='gray')

# %% Get (normalized)feature vectors
# train_feature_vectors = None
# for img in train_images:
#     features = extract_features(img)
#     for f in features:
#          train_feature_vectors.append(f)
# train_feature_vectors = np.array(train_feature_vectors)

train_feature_vectors = None
for img in train_images:
    feature_vectors = extract_features(img)
    if(train_feature_vectors is None):
        train_feature_vectors = feature_vectors
    else:
        train_feature_vectors = np.vstack((train_feature_vectors, feature_vectors))

# %% Clustering
K = 500 # For testing
# kmeans = KMeans(K).fit(train_feature_vectors)
kmeans = MiniBatchKMeans(K, init_size=3*K).fit(train_feature_vectors)
codewords = kmeans.cluster_centers_
labels = kmeans.labels_

# %% Sample visual words
fig, ax = plt.subplots(nrows=1, ncols=20, figsize=(18,3))
for i in range(20):
    ax[i].imshow(np.reshape(codewords[i], (4,4)), cmap='gray')
    ax[i].set_xticks([],[])
    ax[i].set_yticks([],[])

# %% Generate image histogram
def build_img_histogram(img, kmeans):
    features = extract_features(img)
    cluster_predict = kmeans.predict(features)
    histogram = np.histogram(cluster_predict, bins=range(len(kmeans.cluster_centers_) + 1), density=True)[0]
    return histogram

# %% Generate historgrams of train and verified images
train_img_histograms = np.zeros((len(train_images), K))
for i in range(len(train_images)):
    train_img_histograms[i] = build_img_histogram(train_images[i], kmeans)

# %% Sample train histogram
for i in range(NUM_CLASS):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,3))
    hist = train_img_histograms[np.where(np.array(train_image_labels) ==  (i + 1))[0],:]
    for j in range(3):
        ax[j].set_title(class_names[i])
        ax[j].bar(range(0, K), hist[j])

# %% Sample verified histogram
for i in np.random.randint(0, len(verified_images), size=10):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,3))
    ax[0].set_title(class_names[verified_image_classes[i] - 1])
    ax[0].imshow(verified_images[i], cmap='gray')
    ax[1].bar(range(0, K), verified_img_histograms[i])

# %% Build SVM classifiers
classifiers = LinearSVC(random_state=0, tol=1e-3, loss='hinge', C=5)
classifiers.fit(train_img_histograms, train_image_labels)
train_predicted = classifiers.predict(train_img_histograms)

# %% Classify train images
for i in np.random.randint(len(train_images), size=20):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,3))
    actual = class_names[train_image_labels[i] - 1]
    predict = class_names[train_predicted[i] - 1]
    ax[0].set_title(f'{actual}(A) {predict}(P)')
    ax[0].imshow(train_images[i], cmap='gray')
    ax[1].bar(range(0,K), train_img_histograms[i])

correct = 0
for i in range(len(train_images)):
    actual = train_image_labels[i]
    predict = train_predicted[i]
    if(actual == predict): correct += 1

accuracy = (float)(correct) / (float)(len(train_images))
print('Train Accuracy : ', accuracy)

# %% Classify verified images
for i in np.random.randint(len(verified_images), size=20):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,3))
    actual = class_names[verified_image_classes[i] - 1]
    predict = class_names[verified_predict[i] - 1]
    ax[0].set_title(f'{actual}(A) {predict}(P)')
    ax[0].imshow(verified_images[i], cmap='gray')
    ax[1].bar(range(0,K), verified_img_histograms[i])

correct = 0
for i in range(len(verified_images)):
    actual = verified_image_classes[i]
    predict = verified_predict[i]
    if(actual == predict): correct += 1

accuracy = (float)(correct) / (float)(len(train_images))
print('Verified Accuracy : ', accuracy)

# %% KFold Cross Validation to find proper K for KMeans
kfold = KFold(n_splits=5,shuffle=True)
NUM_WORDS = 700
all_train_images = np.asarray(all_train_images)
all_train_image_classes = np.asarray(all_train_image_labels)
for train_indices, verified_indices in kfold.split(all_train_images):
    train = all_train_images[train_indices]
    train_labels = all_train_image_classes[train_indices]
    verified = all_train_images[verified_indices]
    verfified_labels = all_train_image_classes[verified_indices]

    print('Extract Features')
    train_vectors = None
    for img in train:
        feature_vectors = extract_features(img)
        if(train_vectors is None):
            train_vectors = feature_vectors
        else:
            train_vectors = np.vstack((train_vectors, feature_vectors))

    print('Build Vocabularies')
    clusters = MiniBatchKMeans(NUM_WORDS, init_size=3*NUM_WORDS).fit(np.array(train_vectors))
    
    print('Build Histogram')
    train_hist = np.zeros((len(train), NUM_WORDS))
    for i in range(len(train)):
        train_hist[i] = build_img_histogram(train[i], clusters)

    print('Train Classifier')
    clf = LinearSVC(random_state=0, tol=1e-3, loss='hinge', C=5)
    clf.fit(train_hist, train_labels)

    print('Classify test set')
    verified_hist = np.zeros((len(verified), NUM_WORDS))
    for i in range(len(verified)):
        verified_hist[i] = build_img_histogram(verified[i], clusters)
    predict = clf.predict(verified_hist)

    print('Verified Accuracy : ', accuracy_score(verfified_labels, predict))
    print('_____________________________________________________')

# %% Generate historgrams of test images
test_img_histograms = np.zeros((len(test_images), K))
for i in range(len(test_images)):
    test_img_histograms[i] = build_img_histogram(test_images[i], kmeans)

for i in np.random.randint(0, len(test_images), size=15):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,3))
    ax[0].imshow(test_images[i], cmap='gray')
    ax[1].bar(range(0, K), test_img_histograms[i])

# %% Classify test images
test_predicted = classifiers.predict(test_img_histograms)
for i in np.random.randint(0, len(test_images), size = 30):
    fig, ax = plt.subplots()
    predict = class_names[test_predicted[i] - 1]
    ax.set_title(f'{predict}')
    ax.imshow(test_images[i], cmap='gray')

# %%
if os.path.exists('run2.txt'): os.remove('run2.txt')
f = open('run2.txt', 'x')
for i in range(len(test_images)):
    predicted_img_name = class_names[test_predicted[i] - 1]
    f = open('run2.txt', 'a')
    f.write(f'{i}.jpg {predicted_img_name}' + '\n')

# %%
