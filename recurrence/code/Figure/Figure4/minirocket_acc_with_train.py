import sys
sys.path.append('/home/zh/opt/project/pycharmproject/minirocket/code')
from minirocket import fit, transform
from sklearn.linear_model import RidgeClassifierCV
import numpy as np
from sklearn.metrics import accuracy_score
import os
import csv
import time

def minirocket_acc(dataset_name):
    
    # read dataset
    X_Y_training = np.loadtxt(f"""/home/zh/opt/project/pycharmproject/minirocket/recurrence/data/UCR/{dataset_name}/{dataset_name}_TRAIN.tsv""", delimiter='\t')
    X_Y_test = np.loadtxt(f"""/home/zh/opt/project/pycharmproject/minirocket/recurrence/data/UCR/{dataset_name}/{dataset_name}_TEST.tsv""", delimiter='\t')
    
    # assign train and test
    X_training = X_Y_training[:, 1:]
    Y_training = X_Y_training[:, 0]
    X_training = X_training.astype(np.float32)
    Y_training = Y_training.astype(np.int)

    X_test = X_Y_test[:, 1:]
    Y_test = X_Y_test[:, 0]
    X_test = X_test.astype(np.float32)
    Y_test = Y_test.astype(np.int)
    
    # transform and fit
    time_a = time.perf_counter()
    parameters = fit(X_training)
    X_training_transform = transform(X_training, parameters)
    X_test_transform = transform(X_test, parameters)
    time_b = time.perf_counter()
    
    classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
    classifier.fit(X_training_transform, Y_training)
    time_c = time.perf_counter()
    

    # predict
    predictions = classifier.predict(X_test_transform)
    
    # accuracy
    acc = accuracy_score(predictions, Y_test)
    
    return acc, time_b - time_a, time_c - time_a

dirct = '/home/zh/opt/project/pycharmproject/minirocket/recurrence/data/UCR'
files=os.listdir(dirct)
print(sorted(files))

cnt = 0
f = open('minirocket_acc_with_train.csv','w',encoding='utf-8')
csv_writer = csv.writer(f)
for file in sorted(files):
    cnt += 1
    acc, transform_time, transform_time_with_fit = minirocket_acc(file)
    print(str(cnt)+ " " + file + " " + str(acc) + " " + str(transform_time) + " " + str(transform_time_with_fit))
    csv_writer.writerow([file, acc, transform_time, transform_time_with_fit])
f.close()