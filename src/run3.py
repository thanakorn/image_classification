import os.path as osp
import pickle

import cyvlfeat as vlfeat
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from src.utilities import load_images_from_directory, load_images_from_folder


def build_vocabulary(images, vocab_size=200):
    sift_features = []
    for image in images:
        frames, descriptors = vlfeat.sift.dsift(image, step=15, fast=False)
        sift_features.extend(descriptors)

    sift_features = np.array(sift_features).astype(np.float32)
    cluster_centers = vlfeat.kmeans.kmeans(sift_features, vocab_size)
    return cluster_centers


def get_bags_of_sifts(images, vocab):
    features = []
    for image in images:
        histogram = np.zeros(len(vocab))
        frames, descriptors = vlfeat.sift.dsift(image, step=3, fast=True)
        assignments = vlfeat.kmeans.kmeans_quantize(descriptors.astype(np.float32), vocab)
        for assignment in np.nditer(assignments):
            histogram[assignment] += 1
        features.append(histogram)

    return preprocessing.normalize(np.array(features))


def svm_classify(train_image_feats, train_labels, test_image_feats):
    svm = LinearSVC(random_state=0, tol=1e-3, loss='hinge', C=5)
    svm.fit(train_image_feats, train_labels)
    test_labels = svm.predict(test_image_feats)
    return test_labels


# Data path
data_path = osp.join('..', 'resources')
training_path = osp.join(data_path, 'training')
testing_path = osp.join(data_path, 'testing')

# Loading image from path
(train_images, train_image_classes) = load_images_from_directory(training_path)
(test_images, file_names) = load_images_from_folder(testing_path)

# Create vocab and store in a file.
vocab_filename = 'vocab1.pkl'
if not osp.isfile(vocab_filename):
    vocab_size = 200
    vocab = build_vocabulary(train_images, vocab_size)
    with open(vocab_filename, 'wb') as f:
        pickle.dump(vocab, f)
        print('{:s} saved'.format(vocab_filename))

# Use pre-created vocab
with open(vocab_filename, 'rb') as f:
    vocab = pickle.load(f)

train_images_features = get_bags_of_sifts(train_images, vocab)
train_image_classes = np.array(train_image_classes)
test_image_features = get_bags_of_sifts(test_images, vocab)

# prediction_result = svm_classify(train_images_features, train_image_classes, test_image_features)

# Split train/ test data
# Shuffle data
indices = np.arange(train_images_features.shape[0])
np.random.shuffle(indices)
train_images_features = train_images_features[indices]
train_image_classes = train_image_classes[indices]

X_train, X_test, y_train, y_test = train_test_split(train_images_features, train_image_classes, test_size=0.2,
                                                    random_state=42)

prediction_result = svm_classify(X_train, y_train, X_test)
score = accuracy_score(y_test, prediction_result)
print('score: ', score)
