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

def minirocket_arff_acc(dataset_name, sample_num):
    
    # read data
    X_Y_training_path = f"""/home/zh/opt/project/pycharmproject/minirocket/recurrence/data/{dataset_name}/{dataset_name}_TRAIN.arff"""
    X_Y_test_path = f"""/home/zh/opt/project/pycharmproject/minirocket/recurrence/data/{dataset_name}/{dataset_name}_TEST.arff"""
    
    # data format transform arff -> dataframe -> numpy.array
    X_Y_training_data = arff.loadarff(X_Y_training_path)
    X_Y_test_data = arff.loadarff(X_Y_test_path)
    
    X_Y_training_df = pd.DataFrame(X_Y_training_data[0])
    X_Y_training_df = X_Y_training_df.sample(n=sample_num, random_state=1)
    X_Y_training_df['target'] = X_Y_training_df['target'].apply(lambda x : x.decode('utf-8'))
    X_Y_training_df['target'] = X_Y_training_df['target'].map(dict(zip(X_Y_training_df['target'].unique(), range(X_Y_training_df['target'].nunique()))))
    
    X_Y_test_df = pd.DataFrame(X_Y_test_data[0])
    X_Y_test_df = X_Y_test_df.sample(n=sample_num, random_state=1)
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
    time_a = time.perf_counter()
    
    parameters = fit(X_training)
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

def get_sample_num(dataset_name):
    
    if dataset_name == 'InsectSound':
        return [2**i for i in range(13, 15)]
    elif dataset_name == 'FruitFlies':
        return [2**i for i in range(10, 15)]
    elif dataset_name == 'MosquitoSound':
        return [2**i for i in range(10, 18)]
    

# dataset_names = ['InsectSound', 'FruitFlies', 'MosquitoSound']
dataset_names = ['InsectSound']

for dataset_name in dataset_names:
    print(dataset_name)
    sample_num_list = get_sample_num(dataset_name)
    for sample_num in sample_num_list:
        acc, t = minirocket_arff_acc(dataset_name, sample_num)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + str(sample_num) + ' ' + str(acc) + ' ' + str(t))

print("Finish")