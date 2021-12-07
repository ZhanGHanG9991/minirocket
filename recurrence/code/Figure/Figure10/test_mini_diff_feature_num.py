import sys
sys.path.append('/home/zh/opt/project/pycharmproject/minirocket/code')
from minirocket_different_feature_number import fit, transform
from sklearn.linear_model import RidgeClassifierCV
import numpy as np
from sklearn.metrics import accuracy_score
import os
import csv
import time

def minirocket_acc(dataset_name, num_features=10_000, max_dilations_per_kernel=32):
    
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
    
    parameters = fit(X_training, num_features=num_features)
    X_training_transform = transform(X_training, parameters)
    X_test_transform = transform(X_test, parameters)
    
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

feature_nums = [84, 420, 924, 4956, 9996, 49980, 99960]

cnt = 0
f = open('mini_diff_feature_num.csv','w',encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow(['classifier_name', 'dataset_name', 'accuracy'])
for feature_num in feature_nums:
    cnt = 0
    for file in sorted(files):
        cnt += 1
        acc = minirocket_acc(file, feature_num)
        print(str(feature_num) + " " + str(cnt)+ " " + file + " " + str(acc))
        csv_writer.writerow([str(feature_num), file, acc])

f.close()