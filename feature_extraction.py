import os
import cv2
import numpy as np
from sklearn import preprocessing
from skimage.feature import local_binary_pattern
from progress.bar import Bar
import time
from pathlib import Path


def main():
    main_start_time = time.time()
    train_image_path = Path('./Projeto/images_split/train/')
    test_image_path = Path('./Projeto/images_split/val/')
    train_feature_path = Path('./Projeto/features_labels/train/')
    test_feature_path = Path('./Projeto/features_labels/val/')

    process_images(train_image_path, train_feature_path, 'TRAINING')
    process_images(test_image_path, test_feature_path, 'TEST')

    elapsed_time = round(time.time() - main_start_time, 2)
    print(f'[INFO] Code execution time: {elapsed_time}s')


def process_images(image_path, feature_path, data_type):
    print(f'[INFO] ========= {data_type} IMAGES ========= ')
    images, labels = get_data(image_path)
    encoded_labels, encoder_classes = encode_labels(labels)
    #features = extract_LBP_features(images)
    features = extract_hu_moments_features(images)
    save_data(feature_path, encoded_labels, features, encoder_classes)


def get_data(path):
    images = []
    labels = []
    if os.path.exists(path):
        for dirpath, dirnames, filenames in os.walk(path):
            if (len(filenames) > 0):
                folder_name = os.path.basename(dirpath)
                progress_bar = Bar(f'[INFO] Getting images and labels from {folder_name}', max=len(filenames), suffix='%(index)d/%(max)d Duration:%(elapsed)ds')
                for index, file in enumerate(filenames):
                    label = folder_name
                    labels.append(label)
                    full_path = os.path.join(dirpath, file)
                    image = cv2.imread(full_path)
                    images.append(image)
                    progress_bar.next()
                progress_bar.finish()
        return images, np.array(labels, dtype=object)


def extract_hu_moments_features(images):
    bar = Bar('[INFO] Extrating Hu moments features...', max=len(images), suffix='%(index)d/%(max)d  Duration:%(elapsed)ds')
    features_list = []
    for image in images:
        if (np.ndim(image) > 2):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
        moments = cv2.HuMoments(cv2.moments(image)).flatten()
        features_list.append(moments)
        bar.next()
    bar.finish()
    return np.array(features_list, dtype=object)


def extract_LBP_features(images):
    bar = Bar('[INFO] Extracting LBP features...', max=len(images), suffix='%(index)d/%(max)d  Duration:%(elapsed)ds')
    features_list = []
    for image in images:
        if np.ndim(image) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        features_list.append(hist)
        bar.next()
    bar.finish()
    return np.array(features_list, dtype=object)


def encode_labels(labels):
    start_time = time.time()
    print(f'[INFO] Encoding labels to numerical labels')
    encoder = preprocessing.LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    elapsed_time = round(time.time() - start_time, 2)
    print(f'[INFO] Encoding done in {elapsed_time}s')
    return np.array(encoded_labels, dtype=object), encoder.classes_


def save_data(path, labels, features, encoder_classes):
    start_time = time.time()
    print(f'[INFO] Saving data')
    label_filename = 'labels.csv'
    feature_filename = 'features.csv'
    encoder_filename = 'encoder_classes.csv'
    np.savetxt(str(path / label_filename), labels, delimiter=',', fmt='%i')
    np.savetxt(str(path / feature_filename), features, delimiter=',')
    np.savetxt(str(path / encoder_filename), encoder_classes, delimiter=',', fmt='%s')
    elapsed_time = round(time.time() - start_time, 2)
    print(f'[INFO] Saving done in {elapsed_time}s')


if __name__ == "__main__":
    main()
