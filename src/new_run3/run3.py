import os.path as osp
import pickle

import cyvlfeat as vlfeat
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from utilities import load_images_from_directory, load_images_from_folder


def build_vocabulary(images, vocab_size=230):
    sift_features = []
    for image in images:
        frames, descriptors = vlfeat.sift.dsift(image, step=15, size=3, fast=False)
        sift_features.extend(descriptors)

    sift_features = np.array(sift_features).astype(np.float32)
    cluster_centers = vlfeat.kmeans.kmeans(sift_features, vocab_size)
    return cluster_centers


def get_bags_of_sifts(images, vocab):
    features = []
    for image in images:
        histogram = np.zeros(len(vocab))
        frames, descriptors = vlfeat.sift.dsift(image, step=3, size=3
                                                , fast=False)
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

# # Non-linear SVM
# def svm_classify(train_image_feats, train_labels, test_image_feats):
#     svm = SVC(random_state=0, tol=1e-3, C=2, gamma='auto', decision_function_shape='ovo')
#     svm.fit(train_image_feats, train_labels)
#     test_labels = svm.predict(test_image_feats)
#     return test_labels


# Data path
data_path = osp.join('.', 'resources')
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

# X_train, X_test, y_train, y_test = train_test_split(train_images_features, train_image_classes, test_size=0.2,
#                                                     random_state=42)
#
# # ten fold cross validation
# num_sec = 150
# X_std = train_images_features
# y = train_image_classes
# X_1 = X_std[0:num_sec]
# y_1 = y[0:num_sec]
#
# X_2 = X_std[num_sec:2*num_sec]
# y_2 = y[num_sec:2*num_sec]
#
# X_3 = X_std[2*num_sec:3*num_sec]
# y_3 = y[2*num_sec:3*num_sec]
#
# X_4=X_std[3*num_sec:4*num_sec]
# y_4=y[3*num_sec:4*num_sec]
#
# X_5=X_std[4*num_sec:5*num_sec]
# y_5=y[4*num_sec:5*num_sec]
#
# X_6=X_std[5*num_sec:6*num_sec]
# y_6=y[5*num_sec:6*num_sec]
#
# X_7=X_std[6*num_sec:7*num_sec]
# y_7=y[6*num_sec:7*num_sec]
#
# X_8=X_std[7*num_sec:8*num_sec]
# y_8=y[7*num_sec:8*num_sec]
#
# X_9=X_std[8*num_sec:9*num_sec]
# y_9=y[8*num_sec:9*num_sec]
#
# X_10=X_std[9*num_sec:10*num_sec]
# y_10=y[9*num_sec:10*num_sec]
#
#
# #把数据重新组合成十组训练集和测试集
# X_train1 = np.vstack((X_1,X_2,X_3,X_4,X_5,X_6,X_7,X_8,X_9))
# y_train1 = np.hstack((y_1,y_2,y_3,y_4,y_5,y_6,y_7,y_8,y_9))
# X_test1 = X_10
# y_test1 = y_10
#
#
#
# X_train2=np.vstack((X_1,X_2,X_3,X_4,X_5,X_6,X_7,X_8,X_10))
# y_train2=np.hstack((y_1,y_2,y_3,y_4,y_5,y_6,y_7,y_8,y_10))
# X_test2=X_9
# y_test2=y_9
#
# X_train3=np.vstack((X_1,X_2,X_3,X_4,X_5,X_6,X_7,X_9,X_10))
# y_train3=np.hstack((y_1,y_2,y_3,y_4,y_5,y_6,y_7,y_9,y_10))
# X_test3=X_8
# y_test3=y_8
#
# X_train4=np.vstack((X_1,X_2,X_3,X_4,X_5,X_6,X_8,X_9,X_10))
# y_train4=np.hstack((y_1,y_2,y_3,y_4,y_5,y_6,y_8,y_9,y_10))
# X_test4=X_7
# y_test4=y_7
#
# X_train5=np.vstack((X_1,X_2,X_3,X_4,X_5,X_7,X_8,X_9,X_10))
# y_train5=np.hstack((y_1,y_2,y_3,y_4,y_5,y_7,y_8,y_9,y_10))
# X_test5=X_6
# y_test5=y_6
#
# X_train6=np.vstack((X_1,X_2,X_3,X_4,X_6,X_7,X_8,X_9,X_10))
# y_train6=np.hstack((y_1,y_2,y_3,y_4,y_6,y_7,y_8,y_9,y_10))
# X_test6=X_5
# y_test6=y_5
#
# X_train7=np.vstack((X_1,X_2,X_3,X_5,X_6,X_7,X_8,X_9,X_10))
# y_train7=np.hstack((y_1,y_2,y_3,y_5,y_6,y_7,y_8,y_9,y_10))
# X_test7=X_4
# y_test7=y_4
#
# X_train8=np.vstack((X_1,X_2,X_4,X_5,X_6,X_7,X_8,X_9,X_10))
# y_train8=np.hstack((y_1,y_2,y_4,y_5,y_6,y_7,y_8,y_9,y_10))
# X_test8=X_3
# y_test8=y_3
#
# X_train9=np.vstack((X_1,X_3,X_4,X_5,X_6,X_7,X_8,X_9,X_10))
# y_train9=np.hstack((y_1,y_3,y_4,y_5,y_6,y_7,y_8,y_9,y_10))
# X_test9=X_2
# y_test9=y_2
#
# X_train10=np.vstack((X_2,X_3,X_4,X_5,X_6,X_7,X_8,X_9,X_10))
# y_train10=np.hstack((y_2,y_3,y_4,y_5,y_6,y_7,y_8,y_9,y_10))
# X_test10=X_1
# y_test10=y_1
#
#
#
#
#
# prediction_result1 = svm_classify(X_train1, y_train1, X_test1)
# score1 = accuracy_score(y_test1, prediction_result1)
# print('score: ', score1)
#
# prediction_result2 = svm_classify(X_train2, y_train2, X_test2)
# score2 = accuracy_score(y_test2, prediction_result2)
# print('score: ', score2)
#
# prediction_result3 = svm_classify(X_train3, y_train3, X_test3)
# score3 = accuracy_score(y_test3, prediction_result3)
# print('score: ', score3)
#
# prediction_result4 = svm_classify(X_train4, y_train4, X_test4)
# score4 = accuracy_score(y_test4, prediction_result4)
# print('score: ', score4)
#
# prediction_result5 = svm_classify(X_train5, y_train5, X_test5)
# score5 = accuracy_score(y_test5, prediction_result5)
# print('score: ', score5)
#
# prediction_result6 = svm_classify(X_train6, y_train6, X_test6)
# score6 = accuracy_score(y_test6, prediction_result6)
# print('score: ', score6)
#
# prediction_result7 = svm_classify(X_train7, y_train7, X_test7)
# score7 = accuracy_score(y_test7, prediction_result7)
# print('score: ', score7)
#
# prediction_result8 = svm_classify(X_train8, y_train8, X_test8)
# score8 = accuracy_score(y_test8, prediction_result8)
# print('score: ', score8)
#
# prediction_result9 = svm_classify(X_train9, y_train9, X_test9)
# score9 = accuracy_score(y_test9, prediction_result9)
# print('score: ', score9)
#
# prediction_result10 = svm_classify(X_train10, y_train10, X_test10)
# score10 = accuracy_score(y_test10, prediction_result10)
# print('score: ', score10)


prediction_results = svm_classify(train_images_features, train_image_classes, test_image_features)
output_path = osp.join('.', 'run3.txt')
with open(output_path, 'a') as the_file:
    for i in range(prediction_results.size):
        text = file_names[i] + ' ' + prediction_results[i] + '\n'
        the_file.write(text)