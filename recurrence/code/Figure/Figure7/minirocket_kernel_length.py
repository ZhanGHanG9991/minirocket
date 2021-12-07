import sys
sys.path.append('/home/zh/opt/project/pycharmproject/minirocket/code')
from minirocket_kernel_length_util import fit, transform
from sklearn.linear_model import RidgeClassifierCV
import numpy as np
from sklearn.metrics import accuracy_score
import os
import csv
import time

def minirocket_acc(dataset_name, kernel_length, alpha_num):
    
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
    parameters = fit(X_training, kernel_length=kernel_length, alpha_num=alpha_num)
    X_training_transform = transform(X_training, parameters, kernel_length=kernel_length, alpha_num=alpha_num)
    X_test_transform = transform(X_test, parameters, kernel_length=kernel_length, alpha_num=alpha_num)
    
    classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
    classifier.fit(X_training_transform, Y_training)
    

    # predict
    predictions = classifier.predict(X_test_transform)
    
    # accuracy
    acc = accuracy_score(predictions, Y_test)
    
    return acc

dirct = '/home/zh/opt/project/pycharmproject/minirocket/recurrence/data/UCR'
files=os.listdir(dirct)
print(sorted(files))

cnt = 0
f = open('minirocket_kernel_length.csv','w',encoding='utf-8')
csv_writer = csv.writer(f)

kernel_name = ["9{1}", "9{2}", "9{3}", "9{4}"]

for i in range(len(kernel_name)):
    for file in sorted(files):
        cnt += 1
        acc = minirocket_acc(file, 9, i + 1)
        print(str(cnt)+ " " + file + " " + str(acc))
        csv_writer.writerow([kernel_name[i], file, acc])
f.close()