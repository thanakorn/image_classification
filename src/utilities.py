import cv2
import os

def load_training_images():
    return load_images_from_dir('resources/training')

def load_testing_images():
    return load_images_from_folder('resources/testing')

def load_images_from_dir(dir):
    class_id = 0
    training_images = []
    training_image_classes = []
    class_names = []
    for folder in os.listdir(dir):
        class_id += 1
        class_names.append(folder)
        all_images = load_images_from_folder(os.path.join(dir, folder))
        for img in all_images:
            training_images.append(img)
            training_image_classes.append(class_id)        
    return (training_images, training_image_classes, class_names)

def load_images_from_folder(folder):
    images = []
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images