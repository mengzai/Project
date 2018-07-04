#encoding=utf-8
import pandas as pd
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import pylab as pl
# import plot_features
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from datetime import datetime
import argparse
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--method',type=str,default='RF',help='classfication method')
parser.add_argument('--cv_times',type=int,default=5,help='cv times')
parser.add_argument('--seed',type=int,default=1,help='random seed')
parser.add_argument('--under_sample',type=int,default=1,help='under_sample or not')

args = parser.parse_args()

def load_data(path):
    return pd.read_csv(path, error_bad_lines=False)

if __name__ == '__main__':
	#load data
    data_0 = load_data('data_0')
    data_1 = load_data('data_1')
    #fill missing value
    data_1.fillna(-1, inplace = True)
    data_0.fillna(-1, inplace = True)

	#load args
    cv_times = args.cv_times
    under_sample = bool(args.under_sample)
    seed = args.seed
    method = args.method
    times = 1

    print "cv_times:%d, under_sample:%s, seed:%d, method:%s" %(cv_times, str(under_sample), seed, method)
    #set random seed to ensure that the result could be repeated
    np.random.seed(seed)

    #split data
    for i in range(0,cv_times):
        data_0 = data_0.reindex(np.random.permutation(data_0.index))
        data_1 = data_1.reindex(np.random.permutation(data_1.index))

        if under_sample:
            #将好人data_0:376005   坏人 data_1:115824
            under_sample_data_0 = data_0[0:(data_1.shape[0] * times)]
            no_sample_data_0 = data_0[(data_1.shape[0] * times):-1]
            #将好人数据取坏人数据的大小和剩余大小:under_sample_data_0:115823 no_sample_data_0:260180
            # 此处可以改变但是最后要保证真正环境下测试训练1:1
            traindata = under_sample_data_0[1:int(math.floor(under_sample_data_0.shape[0] * 0.7))]
            traindata = traindata.append(data_1[1:int(math.floor(data_1.shape[0] * 0.7))])
            testdata = under_sample_data_0[int(math.floor(under_sample_data_0.shape[0] * 0.7)):-1]
            testdata = testdata.append(data_1[int(math.floor(data_1.shape[0] * 0.7)):-1])
            # print (len(traindata),len(testdata))
            #将训练集和测试集按照7:3比例划分162150, 69492

            index = int(math.floor(data_1.shape[0] * 0.3)) * (int(data_0.shape[0] * 1.0 / data_1.shape[0]) - 1)
            # print (index)
            #测试集的列数   ;保证index<坏人data数量
            testdata = testdata.append(no_sample_data_0[0:index])
            # print (len(testdata))
            # 138984

        else:
			traindata = data_0[0:int(data_0.shape[0]*0.7)]
			traindata = traindata.append(data_1[0:int(data_1.shape[0]*0.7)])
			testdata = data_0[int(data_0.shape[0]*0.7):-1]
			testdata = testdata.append(data_1[int(data_1.shape[0]*0.7):-1])

    print "testdata shape:"+str(testdata.shape)+"\ttraindata shape:"+str(traindata.shape)
    traindata = traindata.as_matrix()
    testdata = testdata.as_matrix()


    shuffle = False
    n_folds = 10
    X=traindata[:,1:]
    y=traindata[:,0]
    X_Submission=testdata[:,1:]

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]
    skf = list(StratifiedKFold(y, n_folds))
    clfs = [GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=3, random_state=0),
            RandomForestClassifier(n_estimators=300, max_features = 'auto', random_state=132, max_depth=15)]
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_Submission.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print j,clf
        dataset_blend_test_j = np.zeros((X_Submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
                X_train = X[train]
                y_train = y[train]
                X_test = X[test]
                y_test = y[test]
                clf.fit(X_train, y_train)
                y_submission = clf.predict_proba(X_test)[:, 1]
                dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict_proba(X_Submission)[:, 1]
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
    print dataset_blend_test
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:, 1]
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    print y_submission
    print "Saving Results."
    tmp = np.vstack([range(1, len(y_submission)+1), y_submission]).T
    np.savetxt(fname='submission.csv', X=tmp, fmt='%d,%0.9f',
        header='MoleculeId,PredictedProbability', comments='')
    print(metrics.classification_report(testdata[:, 0], y_submission, digits=6))
