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
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.svm import SVC
from datetime import datetime
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

    method = 'SVM'

    #select the classifier
    if method == 'GBDT':
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=3, random_state=0)
    if method == 'AB':
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                               n_estimators=300)
    if method == 'KN':
        clf = KNeighborsClassifier(n_neighbors=10,n_jobs=1)
    if method == 'SVM':
        clf =SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    elif method == 'RF':
        clf = RandomForestClassifier(n_estimators=300, max_features = 'auto', random_state=132, max_depth=15)

    #train and test
    clf.fit(traindata[:,1:],traindata[:,0])
    predict = clf.predict_proba(testdata[:,1:])

    #plot and caculate the ks value
    # plot_features.plot_one_feature(predict[:,1], testdata[:,0], 100, './', "plot_ks", 0, 'XXD', 'XXD', 0)
    #print result on test data and train data
    for i in range(0,len(predict)):
        if predict[i][1] <= 0.5:
            predict[i][0] = 0
        else:
            predict[i][0] = 1
    print('On Test:')
    print(metrics.classification_report(testdata[:,0], predict[:,0], digits=6))
    predict = clf.predict(traindata[:,1:])
    print('On Train:')
    print(metrics.classification_report(traindata[:,0], predict, digits=6))

    # clfs：voting的基本模型；这里选择RF和GBDT两种模型。可以后期添加模型
    clfs = [RandomForestClassifier(n_estimators=300, max_features='auto', random_state=132, max_depth=15),
            GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=3, random_state=0),
            KNeighborsClassifier(n_neighbors=10, n_jobs=1)]
    tscore, fscore = 0.0, 0.0
    predict1 = {}
    predict2 = {}
    pre = []

    # 对every data运用不同模型；判断tscore与fscore的大小来vote
    for j, clf1 in enumerate(clfs):
        clf1.fit(traindata[:, 1:], traindata[:, 0])
        predict1[j] = clf1.predict_proba(testdata[:, 1:])
    for i in range((len(testdata[:, 0]))):
        tscore=0
        fscore=0
        # tscore:求predict[i][1] >= 0.5的概率和；
        # fscore:求predict[i][1] <= 0.5的概率和；
        # 用概率和来代表vote的数量这样后期就不用再加入权重值
        for (key, values) in predict1.items():
            if values[i][1] <= 0.5:
                tscore=tscore+values[i][1]
            else:
                fscore=fscore+values[i][1]
                # voting_sample:此为外部输入参数，由于run.sh文件还未更改，这里先设置为1.选为if tscore>=fscore:voting=1, else:voting=0
                # predict1[i][0]:voting之后的结果 voting_sample:   1:true a   0:fause b
                # a:if tscore>=fscore:predict1[i][0]=1, else:predict1[i][0]=0
                # b:if tscore>fscore:predict1[i][0]=1, else:predict1[i][0]=0
        voting_sample = 1
        if voting_sample:
            if tscore >= fscore:
                pre.append(0)
            else:
                pre.append(1)
        else:
            if tscore > fscore:
                pre.append(0)
            else:
                pre.append(1)
    print('On Test after voting:')
    print(metrics.classification_report(testdata[:, 0], pre[:], digits=6))
#    print(metrics.classification_report(traindata[:, 0], pre[:], digits=6))