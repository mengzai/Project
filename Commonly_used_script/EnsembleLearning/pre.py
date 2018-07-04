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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from datetime import datetime
import argparse
import argparse


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
    data_submission = load_data('submission.csv')
    data_0 = load_data('data_0')
    data_1 = load_data('data_1')
    # fill missing value
    data_1.fillna(-1, inplace=True)
    data_0.fillna(-1, inplace=True)

    # load args
    cv_times = args.cv_times
    under_sample = bool(args.under_sample)
    seed = args.seed
    method = args.method
    times = 1

    print "cv_times:%d, under_sample:%s, seed:%d, method:%s" % (cv_times, str(under_sample), seed, method)
    # set random seed to ensure that the result could be repeated
    np.random.seed(seed)

    # split data
    for i in range(0, cv_times):
        data_0 = data_0.reindex(np.random.permutation(data_0.index))
        data_1 = data_1.reindex(np.random.permutation(data_1.index))

        if under_sample:
            # 将好人data_0:376005   坏人 data_1:115824
            under_sample_data_0 = data_0[0:(data_1.shape[0] * times)]
            no_sample_data_0 = data_0[(data_1.shape[0] * times):-1]
            # 将好人数据取坏人数据的大小和剩余大小:under_sample_data_0:115823 no_sample_data_0:260180
            # 此处可以改变但是最后要保证真正环境下测试训练1:1
            traindata = under_sample_data_0[1:int(math.floor(under_sample_data_0.shape[0] * 0.7))]
            traindata = traindata.append(data_1[1:int(math.floor(data_1.shape[0] * 0.7))])
            testdata = under_sample_data_0[int(math.floor(under_sample_data_0.shape[0] * 0.7)):-1]
            testdata = testdata.append(data_1[int(math.floor(data_1.shape[0] * 0.7)):-1])
            # print (len(traindata),len(testdata))
            # 将训练集和测试集按照7:3比例划分162150, 69492

            index = int(math.floor(data_1.shape[0] * 0.3)) * (int(data_0.shape[0] * 1.0 / data_1.shape[0]) - 1)
            # print (index)
            # 测试集的列数   ;保证index<坏人data数量
            testdata = testdata.append(no_sample_data_0[0:index])
            # print (len(testdata))
            # 138984

        else:
            traindata = data_0[0:int(data_0.shape[0] * 0.7)]
            traindata = traindata.append(data_1[0:int(data_1.shape[0] * 0.7)])
            testdata = data_0[int(data_0.shape[0] * 0.7):-1]
            testdata = testdata.append(data_1[int(data_1.shape[0] * 0.7):-1])
    traindata = traindata.as_matrix()
    testdata = testdata.as_matrix()
    data_submission=data_submission.as_matrix()
    print len(testdata[:, 0]),len(data_submission[:,1])
    for i in range(0, len(data_submission[:,1])):
        if data_submission[i][1] <= 0.5:
            data_submission[i][0] = 0
        else:
            data_submission[i][0] = 1
    print(metrics.classification_report(testdata[:, 0], data_submission[:,0], digits=6))