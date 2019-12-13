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
from sklearn.svm import SVC
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from numpy import linalg as LA
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
        feature_vector = feature_vector[::4] # Sampling every 4 px
        feature_vectors.append(feature_vector) 
    return np.asarray(feature_vectors)

# %% Load images
(train_images, train_image_classes, class_names) = load_training_images()
test_images = load_testing_images()

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
        ax[i][j].set_title(class_names[train_image_classes[(i * NUM_IMG_EACH_CLASS) + random_indices[j]] - 1])
        ax[i][j].imshow(train_images[(i * NUM_IMG_EACH_CLASS) + random_indices[j]], cmap='gray')
        ax[i][j].set_xticks([],[])
        ax[i][j].set_yticks([],[])

# %% Get feature vectors and normalize
train_feature_vectors = []
for img in train_images:
    features = extract_features(img)
    for f in features:
         train_feature_vectors.append(f)
# Vector normalization
normalizer = Normalizer().fit(train_feature_vectors)
train_feature_vectors = normalizer.transform(train_feature_vectors)

# %% Clustering
K = 500 # For testing
# kmeans = KMeans(K).fit(train_feature_vectors)
kmeans = MiniBatchKMeans(K, init_size=3*K).fit(train_feature_vectors)
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
    # normalizer = Normalizer().fit(train_feature_vectors)
    # train_feature_vectors = normalizer.transform(train_feature_vectors)
    # scaler = MinMaxScaler()
    features = extract_features(img)
    normalizer = Normalizer().fit(features)
    features =  normalizer.transform(features)
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
for i in range(NUM_CLASS):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,3))
    hist = train_img_histograms[np.where(np.array(train_image_classes) ==  (i + 1))[0],:]
    for j in range(3):
        ax[j].set_title(class_names[i])
        ax[j].bar(range(0, K), hist[j])

# %% Test images histogram
for i in np.random.randint(0, len(test_images), size=5):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,3))
    ax[0].imshow(test_images[i], cmap='gray')
    ax[1].bar(range(0, K), test_img_histograms[i])

# %% Build classifiers
classifiers = SVC()
classifiers.fit(train_img_histograms, train_image_classes)
train_predicted = classifiers.predict(train_img_histograms)
test_predicted = classifiers.predict(test_img_histograms)

# %% Classify train images
for i in np.random.randint(len(train_images), size=20):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,3))
    actual = class_names[train_image_classes[i] - 1]
    predict = class_names[train_predicted[i] - 1]
    ax[0].set_title(f'{actual}(A) {predict}(P)')
    ax[0].imshow(train_images[i], cmap='gray')
    ax[1].bar(range(0,K), train_img_histograms[i])

# %% Accuracy on training set
correct = 0
for i in range(len(train_images)):
    actual = train_image_classes[i]
    predict = train_predicted[i]
    if(actual == predict): correct += 1

accuracy = (float)(correct) / (float)(len(train_images))
print('Accuracy : ', accuracy)

# %% Classify test images
if os.path.exists('run2.txt'): os.remove('run2.txt')
f = open('run2.txt', 'x')
for i in range(len(test_images)):
    plt.figure()
    predicted_img_name = class_names[test_predicted[i] - 1]
    plt.title(predicted_img_name)
    plt.imshow(test_images[i], cmap='gray')
    f = open('run2.txt', 'a')
    f.write(f'{i}.jpg {predicted_img_name}' + '\n')