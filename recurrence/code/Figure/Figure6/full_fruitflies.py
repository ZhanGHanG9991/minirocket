from scipy.io import arff
import pandas as pd
import sys
sys.path.append('/home/zh/opt/project/pycharmproject/minirocket/code')
from minirocket import fit, transform
from sklearn.linear_model import RidgeClassifierCV
import numpy as np
from sklearn.metrics import accuracy_score
import os
import csv
import time

def minirocket_arff_acc(dataset_name):
    
    # read data
    X_Y_training_path = f"""/home/zh/opt/project/pycharmproject/minirocket/recurrence/data/{dataset_name}/{dataset_name}_TRAIN.arff"""
    X_Y_test_path = f"""/home/zh/opt/project/pycharmproject/minirocket/recurrence/data/{dataset_name}/{dataset_name}_TEST.arff"""
    
    # data format transform arff -> dataframe -> numpy.array
    X_Y_training_data = arff.loadarff(X_Y_training_path)
    X_Y_test_data = arff.loadarff(X_Y_test_path)
    
    X_Y_training_df = pd.DataFrame(X_Y_training_data[0])
    X_Y_training_df['target'] = X_Y_training_df['target'].apply(lambda x : x.decode('utf-8'))
    X_Y_training_df['target'] = X_Y_training_df['target'].map(dict(zip(X_Y_training_df['target'].unique(), range(X_Y_training_df['target'].nunique()))))
    
    X_Y_test_df = pd.DataFrame(X_Y_test_data[0])
    X_Y_test_df['target'] = X_Y_test_df['target'].apply(lambda x : x.decode('utf-8'))
    X_Y_test_df['target'] = X_Y_test_df['target'].map(dict(zip(X_Y_test_df['target'].unique(), range(X_Y_test_df['target'].nunique()))))
    
    X_Y_training = X_Y_training_df.values
    X_Y_test = X_Y_test_df.values
    
    X_training = X_Y_training[:, :-1]
    Y_training = X_Y_training[:, -1]
    X_training = X_training.astype(np.float32)
    Y_training = Y_training.astype(np.int)
    
    X_test = X_Y_test[:, :-1]
    Y_test = X_Y_test[:, -1]
    X_test = X_test.astype(np.float32)
    Y_test = Y_test.astype(np.int)
    
    # transform and fit
    
    parameters = fit(X_training)
    time_a = time.perf_counter()
    X_training_transform = transform(X_training, parameters)
    X_test_transform = transform(X_test, parameters)

    classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
    classifier.fit(X_training_transform, Y_training)
    
    time_b = time.perf_counter()
    
    # predict
    predictions = classifier.predict(X_test_transform)
    
    # accuracy
    acc = accuracy_score(predictions, Y_test)
    
    return acc, time_b - time_a


acc, t = minirocket_arff_acc('FruitFlies')
print("FruitFlies")
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + str(acc) + ' ' + str(t))
print("Finish")