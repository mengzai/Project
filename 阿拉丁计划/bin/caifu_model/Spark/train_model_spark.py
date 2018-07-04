#!/usr/bin/env python
# _*_ coding:UTF-8 _*_
# testing
import sys

default_encoding = "utf-8"
if sys.getdefaultencoding() != default_encoding:
	reload(sys)
	sys.setdefaultencoding(default_encoding)

import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.sql import HiveContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import StandardScalerModel, StandardScaler
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.tree import GradientBoostedTrees
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.stat import MultivariateStatisticalSummary
import logging
import time


def field_processing(row):
	label_out = row[0] if row[0] in [1, 0] else 0
	feature_out = []
	for raw_field in row[1:]:
		if raw_field not in [None, "", "NULL", "null"]:
			try:
				feature_out.append(float(raw_field))
			except:
				feature_out.append(-999.9)
		else:
			feature_out.append(-999.9)
	row_out = LabeledPoint(float(label_out), np.array(feature_out))
	return row_out


class SparkJob(object):
	def __init__(self, name="caifuduan_test"):
		self.sc = SparkContext(appName=name)

	def load_data(self):
		hiveCtx = HiveContext(self.sc)
		hiveCtx.sql('use test')
		sql_text = "select * from alading_train_4_10 where datadate < '2016-09-01' and datadate < '2016-10-01'"
		tb = hiveCtx.sql(sql_text)
		return tb

	def feature_extract(self, rdd_data):
		rdd_data = rdd_data.drop("ecif_id").drop("datadate") \
			.drop("ecif_id_activity").drop("datadate_activity") \
			.drop("ecif_id_communicate").drop("datadate_communicate") \
			.drop("ecif_id_kyc").drop("datadate_kyc") \
			.drop("ecif_id_order").drop("datadate_order") \
			.drop("ecif_id_trade").drop("datadate_trade")
		feature_train = rdd_data.map(field_processing)
		return feature_train

	def data_processing(self, rdd_data):
		features = rdd_data.map(lambda row: row.features)
		labels = rdd_data.map(lambda row: row.label)
		scaler = StandardScaler(withMean=True, withStd=True).fit(features)
		scaled_data = labels.zip(scaler.transform(features))
		scaled_data = scaled_data.map(lambda (x, y): LabeledPoint(x, y))
		return scaled_data

	def nb_processing(self, rdd_data):  # negative value trans to 0
		labels = rdd_data.map(lambda row: row.label)
		features = rdd_data.map(lambda row: row.features)
		process_features = features.map(lambda x: [[i, 0][i < 0] for i in x])
		process_data = labels.zip(process_features)
		nb_data = process_data.map(lambda (x, y): LabeledPoint(x, y))
		return nb_data

	def model_save(self, model_list, path):
		for model in model_list:
			print model.__class__.__name__
			model.save(self.sc, path + str(model.__class__.__name__))

	def model_train(self, data_rdd):
		train_data, test_data = data_rdd.randomSplit([0.7, 0.3], 123)
		# 1.模型训练 -train start
		rdd_data = train_data

		lrModel = LogisticRegressionWithSGD.train(rdd_data, iterations=10)
		svmModel = SVMWithSGD.train(rdd_data, iterations=10)
		dtModel = DecisionTree.trainClassifier(rdd_data, numClasses=2, categoricalFeaturesInfo={}, impurity='entropy',
		                                       maxDepth=5, maxBins=32)
		nbModel = NaiveBayes.train(self.nb_processing(rdd_data))
		rfModel = RandomForest.trainClassifier(rdd_data, numClasses=2, categoricalFeaturesInfo={}, numTrees=3,
		                                       featureSubsetStrategy="auto", impurity='entropy', maxDepth=4, maxBins=32)
		gbdtModel = GradientBoostedTrees.trainClassifier(rdd_data, categoricalFeaturesInfo={}, numIterations=3)

		print 'lrModel param:', lrModel
		print 'svmModel param:', svmModel
		print 'dtModel param:', dtModel
		print 'nbModel param:', nbModel
		print 'randomforestModel param:', rfModel
		print 'gbdtModel param:', gbdtModel
		self.model_evalua(test_data, [lrModel, svmModel, dtModel])
		return [lrModel, svmModel, dtModel, nbModel, rfModel, gbdtModel]

	def model_evalua(self, data_rdd, model_list):
		# 2.模型评价 - ACC & AUC & PR
		rdd_data = data_rdd
		lrModel, svmModel, dtModel = model_list
		# 逻辑回归模型&支持向量机模型 - lrModel, svmModel
		evaluate_out = []
		for model in [lrModel, svmModel]:
			scoresAndLabels = rdd_data.map(lambda row: (float(model.predict(row.features)), row.label)).collect()
			rdd_scoresAndLabels = self.sc.parallelize(scoresAndLabels)
			ACC = rdd_scoresAndLabels.map(lambda row: 1 if (row[0] == row[1]) else 0).sum() / (
			rdd_scoresAndLabels.count() * 1.0)
			metrics = BinaryClassificationMetrics(rdd_scoresAndLabels)
			model_evaluate = [model.__class__.__name__, ACC, metrics.areaUnderROC, metrics.areaUnderPR]
			evaluate_out.append(model_evaluate)
		# print '-- %s -- the ACC is：%f ;  the AUC is:%f ;  the PR is :%f'%(model.__class__.__name__, ACC, metrics.areaUnderROC, metrics.areaUnderPR)

		# 决策树模型 - dtModel
		predictLabel = dtModel.predict(rdd_data.map(lambda row: row.features)).collect()
		trueLabel = rdd_data.map(lambda row: row.label).collect()
		scoresAndLabels = zip(predictLabel, trueLabel)
		scoresAndLabels = [[float(i), j] for i, j in scoresAndLabels]
		rdd_scoresAndLabels = self.sc.parallelize(scoresAndLabels)
		ACC = rdd_scoresAndLabels.map(lambda row: 1 if (row[0] == row[1]) else 0).sum() / (
		rdd_scoresAndLabels.count() * 1.0)
		metrics = BinaryClassificationMetrics(rdd_scoresAndLabels)
		evaluate_out.append(model_evaluate)
		#print '-- %s -- the ACC is：%f ;  the AUC is:%f ;  the PR is :%f'%(dtModel.__class__.__name__, ACC ,metrics.areaUnderROC, metrics.areaUnderPR)

		for ii in evaluate_out:
			print '-- %s -- the ACC is：%f ;  the AUC is:%f ;  the PR is :%f' % (ii[0], ii[1], ii[2], ii[3])

	def run(self):
		logging.basicConfig(level=logging.INFO,
		                    format='%(asctime)s %(levelname)s %(filename)s[line:%(lineno)d] : %(message)s',
		                    datefmt='%y/%m/%d %H:%M:%S')
		logging.info(' run_strat ##')

		# 1.加载数据 - train_file
		df_1 = self.load_data()

		# 2.特征提取 - feature in
		df_2 = self.feature_extract(df_1)
		df_2.cache()

		#3.数据处理(规范化数据)  rdd_data->scaled_data
		df_3 = self.data_processing(df_2)
		print 'The scaled data :', df_3.take(3)

		# 4.模型训练 - train start
		self.model_train(df_3)

		## 5.保存模型 - model save
		'''out_dir = "file:///opt/program/work/dev/wenqiangwang5/spark_demo/tmp/"
		self.model_save(model_list,out_dir)'''
		logging.info(" run_over ##")


def main():
	job = SparkJob()
	job.run()


if __name__ == "__main__":
	main()
