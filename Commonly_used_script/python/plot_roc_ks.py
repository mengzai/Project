#encoding=utf-8
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from compiler.ast import flatten



class Plot_KS_ROC():
    def __init__(self, groud_truth, y_pred, save_path):
        self.groud_truth = groud_truth
        self.y_pred = y_pred
        self.save_path=save_path

    def plot_roc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.groud_truth, self.y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        fig = plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, 'r-', linewidth=2.0, aa=False, label='ROC (area=%0.2f)' % (auc))
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
        plt.xlim([0.0, 1.00])
        plt.ylim([0.0, 1.00])
        plt.xlabel('False Postive Rate')
        plt.ylabel('True Postive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        #plt.savefig('%s/roc_curve.png' % (self.save_path), dpi=180)
        fig.savefig('%s/roc_curve.png' % (self.save_path), dpi=180)

    def plot_ks(self):
        groud_truth = np.array(self.groud_truth)
        index = groud_truth != 1
        groud_truth[index] = 0  # if not 1, then transform to 0
        order = np.argsort(self.y_pred)  # sort y_pred
        groud_truth_tmp = list(groud_truth[order])  # sort groud_truth according to y_pred
        groud_truth_tmp.reverse()

        num_bin = 50
        len_bin = int(len(groud_truth_tmp) * 1.0 / num_bin)
        group = [groud_truth_tmp[i * len_bin:(i + 1) * len_bin] for i in range(0, num_bin)]
        group[-1].append(groud_truth_tmp[num_bin * len_bin:])
        group[-1] = flatten(group[-1][:])

        total = len(groud_truth_tmp)
        total_good = sum(groud_truth_tmp)
        total_bad = total - total_good

        good_list = [sum(group[i]) for i in range(0, num_bin)]  # number of good for each group
        bad_list = [len(group[i]) - good_list[i] for i in range(0, num_bin)]  # number of bad for each group
        good_ks_result_list = [0]#accumulated freq for good
        bad_ks_result_list = [0]#accumulated freq for bad

        for i in range(1, num_bin + 1):
            good_ratio = sum(good_list[:i]) * 1.0 / total_good
            bad_ratio = sum(bad_list[:i]) * 1.0 / total_bad
            good_ks_result_list.append(good_ratio)
            bad_ks_result_list.append(bad_ratio)

        diff_list = list(abs(np.array(bad_ks_result_list) - np.array(good_ks_result_list)))
        max_ks_gap_index = diff_list.index(max(diff_list))

        length = len(good_ks_result_list)
        index = range(0, length)
        labels = list(np.array(index) * 1.0 / num_bin)

        fig = plt.figure(figsize=(10, 10))
        axes = fig.gca()
        axes.plot(labels, good_ks_result_list, 'r', linewidth=2, label='bad')
        axes.plot(labels, bad_ks_result_list, 'g', linewidth=2, label='good')
        max_ks_gap_good_value = good_ks_result_list[max_ks_gap_index]
        max_ks_gap_bad_value = bad_ks_result_list[max_ks_gap_index]
        annotate_text_y_index = abs(max_ks_gap_bad_value - max_ks_gap_good_value) / 2 + \
                                min(max_ks_gap_good_value, max_ks_gap_bad_value)
        max_ks_gap_value =max(diff_list)
        xytext_value = str(labels[max_ks_gap_index])
        axes.annotate(xytext_value, xy=(max_ks_gap_index * 1.0 / num_bin, 0),
                      xytext=(max_ks_gap_index * 1.0 / num_bin, 0.05),
                      arrowprops=dict(facecolor='red', shrink=0.05))
        axes.plot([max_ks_gap_index * 1.0 / num_bin, max_ks_gap_index * 1.0 / num_bin],
                  [bad_ks_result_list[max_ks_gap_index], good_ks_result_list[max_ks_gap_index]], linestyle='--',
                  linewidth=2.5)
        axes.annotate(str(round(max_ks_gap_value, 2)), xy=(max_ks_gap_index * 1.0 / num_bin, annotate_text_y_index))
        axes.legend()
        axes.set_title('KS Curve')
        fig.savefig('%s/ks_curve.png' % (self.save_path), dpi=180)
        plt.close(fig)


import string
f = open("result_sn.txt","r")
lines = f.readlines()#读取全部内容
i=0
y=[]
pred=[]

for line in lines:
    line=line.split('\t')
    y.append(string.atoi(line[0]))
    pred.append(string.atof(line[1]))

plot_object=Plot_KS_ROC(y,pred,'/Users/yangsu/Documents/scorecard')
plot_object.plot_roc()
plot_object.plot_ks()












