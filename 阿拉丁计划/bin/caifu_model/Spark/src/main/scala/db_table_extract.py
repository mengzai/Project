#!/usr/bin/env python
# _*_ coding:UTF-8 _*_
# testing
import sys
default_encoding = "utf-8"
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)

from pyspark import SparkContext
from pyspark.sql import HiveContext
from textwrap import dedent
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
#from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.feature import StandardScalerModel,StandardScaler
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

class SparkJob(object):
	def __init__(self, name="data_abnoaml_detection"):
		self.sc = SparkContext(appName=name)

	def feature_clean(row):
		label_out = row[0] if row[0] in [0, 1] else 0
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

	def feature_extract(self, rdd_data):
		feature_train = rdd_data.map(self.feature_clean)
		return feature_train



	def get_data_from_hive(self):
		name_database = "test"  # 库
		select_table = "alading_train_4_10"  # 表
		hiveCtx = HiveContext(self.sc)
		hiveCtx.sql("use " + name_database)
		tables = hiveCtx.sql("select * from %s limit 100;" % (select_table))
		return tables

	def model_train(self, data_rdd):
		train_data, test_data = data_rdd.randomSplit([0.6, 0.4], 123)
		## 1.模型训练 -train start
		rdd_data = train_data
		print "#####################311111##################"
		lrModel = LogisticRegressionWithSGD.train(rdd_data, iterations=10)
		print "dsfasdfdasf######################################"
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

	#  return [lrModel,svmModel,dtModel,nbModel,rfModel,gbdtModel]

	def model_evalua(self, data_rdd, model_list):
		## 2.模型评价 - ACC & AUC & PR
		rdd_data = data_rdd
		lrModel, svmModel, dtModel = model_list
		# 逻辑回归模型&支持向量机模型 - lrModel, svmModel
		for model in [lrModel, svmModel]:
			scoresAndLabels = rdd_data.map(lambda row: (model.predict(row.features), row.label)).collect()
			scoresAndLabels = [[float(i), j] for i, j in scoresAndLabels]
			rdd_scoresAndLabels = self.sc.parallelize(scoresAndLabels)
			ACC = rdd_scoresAndLabels.map(lambda row: 1 if (row[0] == row[1]) else 0).sum() / (
			rdd_scoresAndLabels.count() * 1.0)
			metrics = BinaryClassificationMetrics(rdd_scoresAndLabels)
			print '-- %s -- the ACC is：%f ;  the AUC is:%f ;  the PR is :%f' % (
			model.__class__.__name__, ACC, metrics.areaUnderROC, metrics.areaUnderPR)

		# 决策树模型 - dtModel
		predictLabel = dtModel.predict(rdd_data.map(lambda row: row.features)).collect()
		trueLabel = rdd_data.map(lambda row: row.label).collect()
		scoresAndLabels = zip(predictLabel, trueLabel)
		scoresAndLabels = [[float(i), j] for i, j in scoresAndLabels]
		rdd_scoresAndLabels = self.sc.parallelize(scoresAndLabels)
		ACC = rdd_scoresAndLabels.map(lambda row: 1 if (row[0] == row[1]) else 0).sum() / (
		rdd_scoresAndLabels.count() * 1.0)
		metrics = BinaryClassificationMetrics(rdd_scoresAndLabels)
		print '-- %s -- the ACC is：%f ;  the AUC is:%f ;  the PR is :%f' % (
		dtModel.__class__.__name__, ACC, metrics.areaUnderROC, metrics.areaUnderPR)

	def data_processing(self, rdd_data):
		features = rdd_data.map(lambda row: row.features)
		labels = rdd_data.map(lambda row: row.label)
		scaler = StandardScaler(withMean=True, withStd=True).fit(features)
		scaled_data = labels.zip(scaler.transform(features))
		scaled_data = scaled_data.map(lambda (x, y): LabeledPoint(x, y))
		return scaled_data



	def run(self, output):
		import logging
		logging.basicConfig(level=logging.INFO,
		                    format='%(asctime)s %(levelname)s %(filename)s[line:%(lineno)d] : %(message)s',
		                    datefmt='%y/%m/%d %H:%M:%S')
		logging.info(' run_strat ##')

		#从hive表里面加载数据
		tables=self.get_data_from_hive()


		tables=tables[['label','all_investment','year_investment','maximum_amount','insurance_cnt']]
		feature_train=self.feature_extract(tables)

		df_3 = self.data_processing(feature_train)

		#提取特征及label
		# name_tables=tables.map(lambda row:row.tableName).filter(lambda tab:select_table in tab).collect() #action 执行
		print "#######################"
		print df_3.show()
		print "#########################"

		## 4.模型训练 - train start
		model_list = self.model_train(df_3)
		'''
		## 5.保存模型 - model save
		out_dir = "D:\\wqworkspace/work/dianxiao/"
		out_dir = "file:///opt/program/work/dev/wenqiangwang5/spark_demo/tmp/"
		self.model_save(model_list,out_dir)'''
		logging.info(" run_over ##")

		# OUT_PUT=self.sc.parallelize(tables)
		# OUT_PUT.coalesce(1).saveAsTextFile(output)
		logging.info(" run_over ##")

def main():
	try:
		output = sys.argv[1]
	except:
		print >> sys.stderr, "usage: HiveContext.py {OUTPUT}"
		return

	job = SparkJob()
	job.run(output)


if __name__ == "__main__":
	main()