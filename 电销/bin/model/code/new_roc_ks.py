#encoding=utf-8
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from compiler.ast import flatten
import math

def frange(start, end, step = 1.0):
	while end > start:
		yield start
		start += step

def frange2(start, end, step = 500):
	while end > start:
		yield start
		start += step
	yield end

class plot_roc_ks():
	def __init__(self, groud_truth, y_pred, save_path):
		self.groud_truth = groud_truth
		self.y_pred = y_pred
		self.save_path=save_path



	def plot_roc(self):
		fpr, tpr, thresholds = metrics.roc_curve(self.groud_truth, self.y_pred, pos_label=1)
		auc = metrics.auc(fpr, tpr)
		print 'auc:',auc
		plt.plot(fpr, tpr, 'r--', linewidth=2.0, aa=False, label='ROC (area=%0.2f)' % (auc))
		plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
		plt.xlim([0.0, 1.00])
		plt.ylim([0.0, 1.00])
		plt.xlabel('False Postive Rate')
		plt.ylabel('True Postive Rate')
		plt.title('ROC Curve')
		plt.legend(loc="lower right")
		plt.savefig('%s/roc_curve.png' % (self.save_path), dpi=180)

	def plot_ks(self):
		data = sorted(zip(self.groud_truth, self.y_pred), key=lambda x: x[1])
		good = np.sum(np.array(self.groud_truth) == 1)
		bad = len(self.groud_truth) - good
		good_ratio = [0]
		bad_ratio = [0]

		step = 0.01
		bin_good = 0
		bin_bad = 0
		idx = 0

		for i in frange(step,1,step):
			bin_bad_tmp=bin_bad
			bin_good_tmp=bin_good
			while idx < len(data) and data[idx][1] <= i:
				if data[idx][0] == 1:
					bin_good += 1
				else:
					bin_bad += 1
				idx += 1
			good_ratio.append(bin_good * 1.0 / good)
			bad_ratio.append(bin_bad * 1.0 / bad)
		# for i in frange2(500,len(self.groud_truth)):
		# 	while idx <i:
		# 		if data[idx][0] == 1:
		# 			bin_good += 1
		# 		else:
		# 			bin_bad += 1
		# 		idx += 1
		# 	good_ratio.append(bin_good * 1.0 / good)
		# 	bad_ratio.append(bin_bad * 1.0 / bad)


		max_gap = max(map(lambda x,y: math.fabs(x-y), good_ratio, bad_ratio))
		max_gap_idx = map(lambda x,y: math.fabs(x-y), good_ratio, bad_ratio).index(max_gap)
		x = [ i for i in frange(0, 1, step) ]
		#x = [i*1.0/len(self.groud_truth) for i in frange2(0,len(self.groud_truth))]
		print 'ks:',max_gap, max_gap_idx

		fig = plt.figure(figsize=(10,10)) 
		axes = fig.gca()
		axes.plot(x, good_ratio, 'g', linewidth=2, label='good')
		axes.plot(x, bad_ratio, 'r', linewidth=2, label='bad')

		axes.annotate(str(max_gap_idx*step), xy=(max_gap_idx*step, 0),
				      xytext=(max_gap_idx*step, 0.05),
				      arrowprops=dict(facecolor='red', shrink=0.05))
		axes.plot([max_gap_idx*step, max_gap_idx*step],
				  [bad_ratio[max_gap_idx], good_ratio[max_gap_idx]],
				  'b--', linewidth=0.5)
		axes.annotate(str(round(max_gap, 2)), 
				      xy=(max_gap_idx*step, (bad_ratio[max_gap_idx]+good_ratio[max_gap_idx])/2.0)
				     )

		axes.legend()
		axes.set_title('KS Curve')
		fig.savefig('%s/ks_curve.png' % (self.save_path), dpi=180)
		plt.close(fig)
		
			


if __name__ == '__main__':
	data = map(lambda x: x.strip('\n').split(','), open('res','rb').readlines())
	gt = [float(i[0]) for i in data]
	pred = [float(i[1]) for i in data]
	pltkr = plot_roc_ks(gt, pred, '.')
	pltkr.plot_roc()
	pltkr.plot_ks()






