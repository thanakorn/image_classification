import cv2
import os

def load_training_images():
    return load_images_from_dir('resources/training')

def load_testing_images():
    return load_images_from_folder('resources/testing')

def load_images_from_dir(dir):
    training_images = []
    for folder in os.listdir(dir):
        all_images = load_images_from_folder(os.path.join(dir, folder))
        for img in all_images:
            training_images.append(img)
    return training_images

def load_images_from_folder(folder):
    images = []
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images