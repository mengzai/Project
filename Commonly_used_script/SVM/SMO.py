#coding=utf-8
import random
from numpy import *
import re
import matplotlib.pyplot as plt


def loadDataSet(fileName):
	MAT=[]
	LabelMat=[]
	fr = open(fileName)
	for line in fr.readlines():

		lineArr=line.strip('\n').split('\t')
		arr= [float(lineArr[0]), float(lineArr[1])]
		MAT.append(arr)
		LabelMat.append(float(lineArr[2]))
	return MAT,LabelMat

#随机选择向量
def selectJrand(i,m):
	j=i
	while (j==i):
	    j=int(random.uniform(0,m))
	return j

def clipAlpha(aj,H,L):
	if aj>H:
	    aj=H
	if L>aj:
	    aj=L
	return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
	dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
	b = 0; m,n = shape(dataMatrix)
	alphas = mat(zeros((m,1)))
	iter = 0
	while (iter < maxIter):
	    alphaPairsChanged = 0
	    for i in range(m):
			# yi(wtx+b-yi)<-$  alphas[i]<C or  yi(wtx+b-yi)>-$  alphas[i]>0   ????????????
	        fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
	        Ei = fXi - float(labelMat[i])
			# yi(wtx+b-yi)<-$  alphas[i]<C or  yi(wtx+b-yi)>-$  alphas[i]>0   ????????????
	        if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
	            j = selectJrand(i,m)
	            fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
	            Ej = fXj - float(labelMat[j])
	            alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
	            if (labelMat[i] != labelMat[j]):
	                L = max(0, alphas[j] - alphas[i])
	                H = min(C, C + alphas[j] - alphas[i])
	            else:
	                L = max(0, alphas[j] + alphas[i] - C)
	                H = min(C, alphas[j] + alphas[i])

	            if L==H: print "L==H"; continue
	            eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
	            if eta >= 0: print "eta>=0"; continue
	            alphas[j] -= labelMat[j]*(Ei - Ej)/eta
	            alphas[j] = clipAlpha(alphas[j],H,L)
	            if (abs(alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; continue
	            alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
	            b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
	            b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
	            if (0 < alphas[i]) and (C > alphas[i]): b = b1
	            elif (0 < alphas[j]) and (C > alphas[j]): b = b2
	            else: b = (b1 + b2)/2.0
	            alphaPairsChanged += 1
	            print("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
	    if (alphaPairsChanged == 0): iter += 1
	    else: iter = 0
	    print("iteration number: %d" % iter)
	return b,alphas

def smo():
	dataMat,labelMat=loadDataSet('testSet.txt')
	fig=plt.figure()
	ax=fig.add_subplot(111)

	for i in range(len(dataMat)):
		ax.scatter(dataMat[i][0], dataMat[i][1], s=30, c='green', marker='s')
	b,alphas=smoSimple(dataMat,labelMat,0.6,0.001,40)

	for i in range(len(dataMat)):
		if alphas[i]>0:
		 ax.scatter(dataMat[i][0], dataMat[i][1], s=50, c='red', marker='s')

	plt.show()

if __name__ == '__main__':
    smo()