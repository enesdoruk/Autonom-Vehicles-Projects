import os
import pickle
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from util.config import root_data_non_vehicle, root_data_vehicle, feat_extraction_params
from util.functions_detection import draw_boxes
from util.functions_detection import search_windows
from util.functions_detection import slide_window
from util.feature_extraction import extract_features_from_file_list
from util.utils import get_file_list_recursively


if __name__ == '__main__':

    cars = get_file_list_recursively(root_data_vehicle)
    notcars = get_file_list_recursively(root_data_non_vehicle)

    print('Extracting car features...')
    car_features = extract_features_from_file_list(cars, feat_extraction_params)

    print('Extracting non-car features...')
    notcar_features = extract_features_from_file_list(notcars, feat_extraction_params)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    feature_scaler = StandardScaler().fit(X) 
    scaled_X = feature_scaler.transform(X)

    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Feature vector length:', len(X_train[0]))

    svc = LinearSVC()  

    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')

    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    with open('data/svm_trained.pickle', 'wb') as f:
        pickle.dump(svc, f)
    with open('data/feature_scaler.pickle', 'wb') as f:
        pickle.dump(feature_scaler, f)
    with open('data/feat_extraction_params.pickle', 'wb') as f:
        pickle.dump(feat_extraction_params, f)
