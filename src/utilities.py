import fnmatch
import os

import cv2


def load_training_images():
    return load_images_from_dir('/Users/mark/Projects/image_classification/resources/training')


def load_testing_images():
    return load_images_from_folder('/Users/mark/Projects/image_classification/resources/testing')


def load_images_from_dir(dir):
    class_id = 0
    training_images = []
    training_image_classes = []
    class_names = []
    for folder in os.listdir(dir):
        class_id += 1
        class_names.append(folder)
        if os.path.isdir(os.path.join(dir, folder)):
            all_images, file_names = load_images_from_folder(os.path.join(dir, folder))
            for img in all_images:
                training_images.append(img)
                training_image_classes.append(class_id)
    return training_images, training_image_classes, class_names


def load_images_from_directory(dir):
    training_images = []
    training_image_classes = []
    for folder in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, folder)):
            all_images, file_names = load_images_from_folder(os.path.join(dir, folder))
            for img in all_images:
                training_images.append(img)
                training_image_classes.append(folder.lower())
    return training_images, training_image_classes


def load_images_from_folder(folder):
    images = []
    file_names = []
    # filter only .jpg file and sort according to image number
    file_list = fnmatch.filter(os.listdir(folder), '*.jpg')
    sorted_file_list = sorted(file_list, key=lambda x: int(os.path.splitext(x)[0]))
    for file in sorted_file_list:
        img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            file_names.append(file)
            images.append(img)
    return images, file_names
