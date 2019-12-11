import os.path as osp

import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

from src.utilities import load_images_from_directory, load_images_from_folder


def get_tiny_image_feature(image, size=(16, 16)):
    crop_image = crop_center(image)
    tiny_image_vector = cv2.resize(crop_image, size).flatten()
    normed_vector = (tiny_image_vector - tiny_image_vector.mean(axis=0)) / tiny_image_vector.std(axis=0)
    return normed_vector


def crop_center(image):
    y, x = image.shape
    if y == x:
        return image
    else:
        crop_size = x if y >= x else y
        start_x = x // 2 - (crop_size // 2)
        start_y = y // 2 - (crop_size // 2)
        return image[start_y:start_y + crop_size, start_x:start_x + crop_size]


data_path = osp.join('..', 'resources')
training_path = osp.join(data_path, 'training')
testing_path = osp.join(data_path, 'testing')

(train_images, train_image_classes) = load_images_from_directory(training_path)
(test_images, file_names) = load_images_from_folder(testing_path)


train_image_tiny = []
for train_image in train_images:
    tiny_image = get_tiny_image_feature(train_image)
    train_image_tiny.append(tiny_image)

test_images_tiny = []
for test_image in test_images:
    tiny_image = get_tiny_image_feature(test_image)
    test_images_tiny.append(tiny_image)

train_image_tiny = np.array(train_image_tiny)
train_image_classes = np.array(train_image_classes)
test_images_tiny = np.array(test_images_tiny)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(train_image_tiny, train_image_classes)
prediction_results = model.predict(test_images_tiny)

for i in range(prediction_results.size):
    print(file_names[i], prediction_results[i])


# Run K-fold cross validation to find optimal number of K
# Shuffle data
# indices = np.arange(tiny_image_feature.shape[0])
# np.random.shuffle(indices)
# tiny_image_feature = tiny_image_feature[indices]
# train_image_classes = train_image_classes[indices]
#
# splits = 10
# scores = {}
# k_list = range(1, 20)
# kf = KFold(n_splits=splits)
# for train_index, test_index in kf.split(tiny_image_feature):
#     X_train, X_test, y_train, y_test = tiny_image_feature[train_index], tiny_image_feature[test_index], \
#                                        train_image_classes[train_index], train_image_classes[test_index]
#
#     for k in k_list:
#         model = KNeighborsClassifier(k)
#         model.fit(X_train, y_train)
#         prediction_result = model.predict(X_test)
#         score = accuracy_score(y_test, prediction_result)
#         scores[k] = scores.get(k, 0) + score
#
# for k in k_list:
#     print(k, scores[k] / splits)
