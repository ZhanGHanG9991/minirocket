import sys
sys.path.append('/home/zh/opt/project/pycharmproject/minirocket/code')
import schedule
import datetime, time
from minirocket_dv import fit_transform
from minirocket import transform
from sklearn.linear_model import RidgeClassifierCV
import numpy as np
from sklearn.metrics import accuracy_score
import os
import csv

def minirocket_dv_acc(dataset_name):
    
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
    parameters, X_training_transform = fit_transform(X_training)

    classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
    classifier.fit(X_training_transform, Y_training)

    X_test_transform = transform(X_test, parameters)

    predictions = classifier.predict(X_test_transform)
    
    return accuracy_score(predictions, Y_test)



def auto_minirocket_dv_acc():
    print('auto_minirocket_dv_acc start')
    dirct = '/home/zh/opt/project/pycharmproject/minirocket/recurrence/data/UCR'
    files=os.listdir(dirct)
    cnt = 0
    f = open('minirocket_dv_acc','w',encoding='utf-8')
    csv_writer = csv.writer(f)
    for file in sorted(files):
        cnt += 1
        acc = minirocket_dv_acc(file)
        print(str(cnt)+ " " + file + " " + str(acc))
        csv_writer.writerow([file, acc])
    f.close()
    print('auto_minirocket_dv_acc end')

def job():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' start')
    auto_minirocket_dv_acc()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' end')

def run():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' run start')
    run_time = '00:37'
    schedule.every().day.at(run_time).do(job)
    while True:
        schedule.run_pending()
        time.sleep(60)

run()

