#encoding=utf-8
import pandas as pd
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
import pylab as pl
# import plot_features
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
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

X_train, X_test, y_train, y_test=traindata[:,1:],testdata[:,1:],traindata[:,0],testdata[:,0]

#生成训练集测试集;

tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)

# 生成500个决策树，详细的参数建议参考官方文档
bag = BaggingClassifier(base_estimator=tree, n_estimators=500, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=1, random_state=1)

# 度量单个决策树的准确性
from sklearn.metrics import accuracy_score
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))
# Output：Decision tree train/test accuracies 1.000/0.854

# 度量bagging分类器的准确性
bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
print('Bagging train/test accuracies %.3f/%.3f' % (bag_train, bag_test))
# Output：Bagging train/test accuracies 1.000/0.896

x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(8, 3))

for idx, clf, tt in zip([0, 1], [tree, bag], ['Decision Tree', 'Bagging']):
    clf.fit(X_train, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], c='blue', marker='^')
    axarr[idx].scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c='red', marker='o')
    axarr[idx].set_title(tt)
plt.show()
