# coding: UTF-8
from __future__ import division
import scipy as sp
import pandas as pd
from weakclassify import WEAKC
import numpy as np
import time
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
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
def sign(x):
    q=np.zeros(np.array(x).shape)
    q[x>=0]=1
    q[x<0]=-1
    return q



# print self.W
def train(M=3):
    '''
        M is the maximal Weaker classification
    '''
    G = {}                                  #G为初级分类器  字典
    alpha = {}                             #G1(x)的系数   字典
    W = np.ones(len(traindata[:,0])).flatten(1) / len(traindata[:,0])  # #将每条数据附权值W1i=0.1    D1=(W1i,W2i.....)
    for i in range(M):
        G.setdefault(i)                     #循环设置第i个基分类器的字典值
        alpha.setdefault(i)
    for i in range(M):
        predict=[]
        err=0
        if i==0:
            clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=3, random_state=0)
        if i==1:
            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                                     n_estimators=300)
        if i==2:
            clf = RandomForestClassifier(n_estimators=300, max_features='auto', random_state=132, max_depth=15)

        clf.fit(X, y)
        predict = clf.predict_proba(X)


        for j in range(len(predict[:,0])):
            if predict[j][1] <= 0.5:
                predict[j][0] = 0
            else:
                predict[j][0] = 1
            if predict[j][0] != y[j]:
                err=err+W[j]

        alpha[i] = 1 / 2 * np.log((1 - err) / err)  # 计算G1 的系数
        print alpha
        Z = W * np.exp(-alpha[i] * y * predict[:, 0])  # 规范化因子
        W = (Z / Z.sum()).flatten(1)  # 更新训练集的权值分布
        print W
        if i == M-1:
            np.savetxt(fname='sub2.csv', X=W )
        if i == M -2:
            np.savetxt(fname='sub1.csv', X=W)
        if i == M - 3:
            np.savetxt(fname='sub0.csv', X=W)
        print finalclassifer(i, sums, predict[:, 0], alpha)
        if finalclassifer(i, sums, predict[:, 0], alpha) == 0:
            print i + 1, " weak classifier is enough to  make the error to 0"
            break


        # G[i] = Weaker(X, y,predict[:,0])

        # clf=GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=3, random_state=0)
        # sg=clf.predict(X)
        # # sg = G[i].pred(X)                      #得到分类器预测的值
        # print sg
        # Z = W * np.exp(-alpha[i] * y * sg.transpose()) #规范化因子
        # W = (Z / Z.sum()).flatten(1)            #更新训练集的权值分布
        # print W
        # Q = i                                   #第i个基分类器
        # # print self.finalclassifer(i),'==========='
        #
        # if finalclassifer(i,sums,G,alpha) == 0:
        #     print i + 1, " weak classifier is enough to  make the error to 0"
        #     break


def finalclassifer(t,sums,Pre,alpha):
    for i in range(t+1):
    	sums=sums+Pre.flatten(1)*alpha[i]
    pre_y=sign(sums)
    t = (pre_y != y).sum()
    return t

if __name__ == '__main__':
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
    y = traindata[:,0]
    X = traindata[:,1:]                # 将矩阵转化为数组
    y = np.array(y).flatten(1)       # 将结果转化为数组当时不是以集合的形式,将list的数据提取出来
    # assert X.shape[1] == y.size
    sums = np.zeros(y.shape)         # 判断多少条数据;
    Q = 0
    Weaker = WEAKC
    train(M=3)
