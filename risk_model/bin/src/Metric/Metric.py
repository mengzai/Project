#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import math
from sklearn.metrics import roc_auc_score
from RiskIndex import RiskIndex
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
import os
from pandas import DataFrame

class Metric(object):
	'''
	功能：模型评价类：主要用来计算模型的KS值，AUC值，以及TOPN统计指标(TOPN样本个数、好样本的个数以及好样本提升指数)
		  以及申请风险评分分布表等指标（风险评分，边际好坏比率，个数和百分比，累计好坏比以及累计坏的概率）
	'''

	def __init__(self, groud_truth, y_pred):
		self.groud_truth = np.array(groud_truth)
		self.y_pred = np.array(y_pred)

	def frange(self, start, end, step=1.0):
		'''
		功能：产生序列
		:param start:开始数
		:param end: 结束数
		:param step: 步长
		:return: 数列
		'''
		while end > start:
			yield start
			start += step

	def calcAuc(self):
		
		fpr, tpr, thresholds = sklearn.metrics.roc_curve(self.groud_truth, self.y_pred, pos_label=1)
		auc = sklearn.metrics.auc(fpr, tpr)
		plt.plot(
			fpr,
			tpr,
			'r--',
			linewidth=2.0,
			aa=False,
			label='ROC(area=%0.2f)' %
				  (auc))
		plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
		plt.xlim([0.0, 1.00])
		plt.ylim([0.0, 1.00])
		plt.xlabel('False Postive Rate')
		plt.ylabel('True Postive Rate')
		plt.title('ROC Curve')
		plt.legend(loc="lower right")
		plt.savefig(os.path.join("./", 'roc_curve.png'), dpi=180)
		return roc_auc_score(self.groud_truth, self.y_pred)
		# plt.close()


	def calcKs(self):
		'''
		功能:计算ks指标
		:param groud_truth: 实际的标签列表
		:param y_pred: 预测的标签列表
		:return: 返回ks指标
		'''
		data = sorted(zip(self.groud_truth, self.y_pred), key=lambda x: x[1])
		good = np.sum(np.array(self.groud_truth) == 1)
		bad = len(self.groud_truth) - good
		good_ratio = [0]
		bad_ratio = [0]
		step = 0.01
		bin_good = 0
		bin_bad = 0
		idx = 0
		for i in self.frange(step, 1 + step, step):
			while idx < len(data) and data[idx][1] <= i:
				if data[idx][0] == 1:
					bin_good += 1
				else:
					bin_bad += 1
				idx += 1
			good_ratio.append(bin_good * 1.0 / good)
			bad_ratio.append(bin_bad * 1.0 / bad)
		max_gap = max(map(lambda x, y: math.fabs(x - y), good_ratio, bad_ratio))
		max_gap_idx = map(
			lambda x,
				   y: math.fabs(
				x - y),
			good_ratio,
			bad_ratio).index(max_gap)
		x = [i for i in self.frange(0, 1.01, step)]
		fig = plt.figure(figsize=(10, 10))
		axes = fig.gca()
		axes.plot(x, good_ratio, 'g', linewidth=2, label='good')
		axes.plot(x, bad_ratio, 'r', linewidth=2, label='bad')
		axes.annotate(str(max_gap_idx * step), \
					  xy=(max_gap_idx * step, 0), \
					  xytext=(max_gap_idx * step, 0.05), \
					  arrowprops=dict(facecolor='red', shrink=0.05))
		axes.plot([max_gap_idx * step,
				   max_gap_idx * step],
				  [bad_ratio[max_gap_idx],
				   good_ratio[max_gap_idx]],
				  'b--',
				  linewidth=0.5)
		axes.annotate(str(round(max_gap, 2)), xy=(
			max_gap_idx * step, (bad_ratio[max_gap_idx] + good_ratio[max_gap_idx]) / 2.0))
		axes.legend()
		axes.set_title('KS Curve')
		fig.savefig(os.path.join("./", 'ks_curve.png'), dpi=180)
		# plt.show()
		plt.close(fig)
		return max_gap

	def calcTopN(self, topN):
		'''
		功能:计算topN样本个数，好样本个数，好样本占比以及好样本提升指数
		:param topN:
		:param groud_truth:
		:param y_pred:
		:return:
		'''
		data_len = len(self.groud_truth)
		N = int(data_len * topN * 1.0 / 100)
		if N<=0:
			return
		sorted_index = np.argsort(-self.y_pred)
		# sorted_y_pred=y_pred[sorted_index]
		sorted_groud_truth = self.groud_truth[sorted_index]
		good_example = 0
		for i in range(N):
			if sorted_groud_truth[i] > 0.5:
				good_example += 1
		good_ratio = good_example * 1.0 / N
		total_good_ratio = np.sum(self.groud_truth) * 1.0 / len(self.groud_truth)
		return N, good_example, good_ratio, good_ratio / total_good_ratio

	def calcRiskScore(self):
		'''
		功能：计算风险模型评分统计指标：
		   具体包括：风险评分，边际好坏率，个数和百分比，累计好坏比，累计坏的概率等统计指标
		:param groud_truth:
		:param y_pred:
		:return:
		'''
		sorted_groud_truth = self.groud_truth[np.argsort(-self.y_pred)]
		split_len = math.ceil(len(self.groud_truth) * 1.0 / 10)
		i = 0
		rank = 0
		summary_list = []
		while i < len(self.groud_truth):
			start = i
			if start + split_len < len(self.groud_truth):
				end = start + split_len
			else:
				end = len(self.groud_truth)
			tmp = sorted_groud_truth[start:end]
			'''计算边际好坏比率'''
			marginal_good_vs_bad_ratio = np.sum(tmp) * 1.0 / (len(tmp)-np.sum(tmp))
			'''计算好的申请者个数和百分比'''
			good_cnt = np.sum(sorted_groud_truth[start:-1])  # 好的个数
			good_precent = good_cnt * 1.0 / \
				np.sum(sorted_groud_truth) * 100  # 好的个数的占比
			'''计算坏的申请者个数和百分比'''
			bad_cnt = len(sorted_groud_truth[start:-1]) - \
				np.sum(sorted_groud_truth[start:-1])  # 坏的个数
			bad_precent = bad_cnt * 1.0 / \
				(len(sorted_groud_truth) - np.sum(sorted_groud_truth)) * 100  # 坏的个数的占比
			'''计算总体的个数和百分比'''
			total_cnt = len(sorted_groud_truth[start:-1])  # 总体的个数
			total_percent = total_cnt * 1.0 / \
				len(sorted_groud_truth) * 100  # 总体的百分比
			'''计算累计好坏比例'''
			cum_good_vs_bad_ratio = good_cnt * 1.0 / bad_cnt
			'''计算累计坏的比例'''
			cum_bad_ratio = bad_cnt * 1.0 / total_cnt
			riskIndex=RiskIndex(rank,marginal_good_vs_bad_ratio,good_cnt,good_precent,bad_cnt,bad_precent,total_cnt,total_percent,cum_good_vs_bad_ratio,cum_bad_ratio)
			rank += 1
			i = end
			summary_list.append([rank,marginal_good_vs_bad_ratio,good_cnt,good_precent,bad_cnt,bad_precent,total_cnt,total_percent,cum_good_vs_bad_ratio,cum_bad_ratio])
		dd=DataFrame(summary_list,columns=['rank','marginal_good_vs_bad_ratio','good_cnt','good_precent','bad_cnt','bad_precent','total_cnt','total_percent','cum_good_vs_bad_ratio','cum_bad_ratio'])
		return dd
