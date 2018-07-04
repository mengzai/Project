#-*- coding: UTF-8 -*-
import csv
import argparse
import os
import pandas as pd
import numpy as np
import numpy
import matplotlib.pyplot as plt
from compiler.ast import flatten
import scipy.stats as stats
import datetime

def plot_ks(groud_truth,y_pred,save_path):
    groud_truth = np.array(groud_truth)
    index = groud_truth != 1
    groud_truth[index] = 0  # if not 1, then transform to 0
    order = np.argsort(y_pred)  # sort y_pred
    groud_truth_tmp = list(groud_truth[order])  # sort groud_truth according to y_pred
    groud_truth_tmp.reverse()

    num_bin = 100
    len_bin = int(len(groud_truth_tmp) * 1.0 / num_bin)
    group = [groud_truth_tmp[i * len_bin:(i + 1) * len_bin] for i in range(0, num_bin)]
    group[-1].append(groud_truth_tmp[num_bin * len_bin:])
    group[-1] = flatten(group[-1][:])

    total = len(groud_truth_tmp)
    total_good = sum(groud_truth_tmp)
    total_bad = total - total_good

    good_list = [sum(group[i]) for i in range(0, num_bin)]  # number of good for each group
    bad_list = [len(group[i]) - good_list[i] for i in range(0, num_bin)]  # number of bad for each group
    good_ks_result_list = [0]  # accumulated freq for good
    bad_ks_result_list = [0]  # accumulated freq for bad

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
    max_ks_gap_value = max(diff_list)
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
    fig.savefig('%s/ks_curve.png' % (save_path), dpi=180)
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    file=open("logistic_regression/test_result.txt")
    groud_truth=[]
    y_pred=[]
    for line in file.readlines():
        line = line.strip()
        line = line.split("\t")
        groud_truth.append(float(line[0]))
        y_pred.append(float(line[2]))

    plot_ks(groud_truth,y_pred,"logistic_regression")
    # plot_ks_score(y_pred, groud_truth, 50, "logistic_regression", 0, title_name='test')

