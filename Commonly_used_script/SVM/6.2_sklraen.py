#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

from sklearn import tree
def model_plot(X,y,h,svc, rbf_svc):
	# create a mesh to plot in
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
						 np.arange(y_min, y_max, h))

	# title for the plots
	titles = ['SVC with linear kernel',
			  'SVC with RBF kernel']

	for i, clf in enumerate((svc, rbf_svc)):
		# Plot the decision boundary. For that, we will assign a color to each
		# point in the mesh [x_min, m_max]x[y_min, y_max].
		plt.subplot(1, 2, i + 1)
		plt.subplots_adjust(wspace=0.4, hspace=0.4)

		Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

		# Put the result into a color plot
		Z = Z.reshape(xx.shape)
		plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

		# Plot also the training points
		plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
		plt.xlabel('Sugar content')
		plt.ylabel('Density')
		plt.xlim(xx.min(), xx.max())
		plt.ylim(yy.min(), yy.max())
		plt.xticks(())
		plt.yticks(())
		plt.title(titles[i])

	plt.show()

def SVM_model(X,y):
	h = .02  # step size in the mesh

	C = 1000  # SVM regularization parameter
	svc=svm.SVC(kernel='linear', C=C).fit(X,y)
	rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)


	return svc,rbf_svc,h

def get_data():
	file=open('watermelon3_0_Ch.csv','rb')
	data=[line.strip('\n').strip('\r').split(',') for line in  file.readlines()]
	data = np.array(data)
	X=[[float(row[-2]),float(row[-1])] for row in data[1:,1:-1] ]
	y=[1 if row[-1]=='1' else 0 for row in data[1:]]

	X=np.array(X)
	y=np.array(y)
	return  X, y

def Iris():
	iris = datasets.load_iris()

	X= iris['data'][iris['target']!=2]
	y=iris['target'][iris['target']!=2]
	return X,y

def watermelon():
	X, y=get_data()

	svc, rbf_svc, h = SVM_model(X, y)
	model_plot(X, y, h, svc, rbf_svc)

def SVM_BP_CART():
	X, y = Iris()
	X=X[:,0:2]
	svc, rbf_svc, h = SVM_model(X, y)
	model_plot(X, y, h, svc, rbf_svc)


def sugar_data_model(X, y):
	C = 1000  # SVM regularization parameter
	svc = svm.SVR(kernel='linear', C=C).fit(X, y)
	rbf_svc = svm.SVR(kernel='rbf', gamma=0.7, C=C).fit(X, y)
	for i, clf in enumerate((svc, rbf_svc)):
		plt.subplot(1, 2, i + 1)
		plt.subplots_adjust(wspace=0.4, hspace=0.4)
		clf.fit(X, y)
		predict_y=clf.predict(X[2])
		predict=clf.predict(X)
		clf.score(X, y)
		plt.scatter(predict,y,s=2)
		plt.plot(predict_y, predict_y, 'ro')
		plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
		plt.xlabel('Predicted')
		plt.ylabel('Measured')
	plt.show()

def get_sugar_data():
	file=open('watermelon3_0_Ch.csv','rb')
	data=[line.strip('\n').strip('\r').split(',') for line in  file.readlines()]
	data = np.array(data)
	X=[[float(row[-2])] for row in data[1:,1:-1] ]
	y= [[float(row[-1])] for row in data[1:, 1:-1]]

	X=np.array(X)
	y=np.array(y)
	return  X, y

def density_to_sugar():
	X, y=get_sugar_data()
	sugar_data_model(X, y)

if __name__ == '__main__':
	SVM_BP_CART()
	watermelon()
	density_to_sugar()