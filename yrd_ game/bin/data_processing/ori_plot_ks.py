#encoding=utf-8
from __future__ import division
from matplotlib import rcParams
import os
from matplotlib.font_manager import fontManager
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import os
import os.path
from matplotlib.lines import Line2D
import argparse
import numpy as np
import argparse
import pandas as pd
import math
import numpy
import matplotlib.pyplot as plt

#rcParams['font.family'] = 'SimSun'


"""
有三个参数必须设置
1) data_path : 指定数据路径
2) has_label : 指定特征的起始位置. 如果数据中有label, 则has_label=2; 如果没有label, 则has_label=1
3) fun_type : 指定需要绘制哪些指标. 备选 ("plot_distribution", "plot_ks", "plot_odds", "plot_gini", "plot_divergence")

注意:
1) fig_save_path 这个最好设置, 不然图像默认放到当前路径的fig/目录下. 如果图像很多不容易归档
2) feature_file feature_name不可同时设置
3) has_label=1的时候fun_type只能选择"plot_distribution"



举例:
1) 绘制全部特征(不设置feature_file和feature_name)
python plot_features.py --data_path datasets1126/all_baoxiao_safe_1 --has_label 2 --fun_type "plot_distribution"  --fig_save_path "fig/fig_baoxiao_safe_1"

2) 绘制指定feature_file 中的特征
python plot_features.py --data_path datasets1126/all_baoxiao_safe_1 --has_label 2 --fun_type "plot_distribution" --feature_file "fea_file" --fig_save_path "fig/fig_baoxiao_safe_1"

3) 绘制指定feature_name 的特征
python plot_features.py --data_path datasets1126/33w --has_label 1 --fun_type "plot_distribution" --feature_name "co_days" --fig_save_path "fig/fig_baoxiao_safe_1"

"""

parser = argparse.ArgumentParser(description="plot features")
parser.add_argument('--data_path', type=str, default='data', required=False, help='the path of data set')
parser.add_argument('--has_label', type=int, default=2, required=False, help='the start index of features in dataset(if no labels in dataset, this is 1; else this is 2)')
#parser.add_argument('--has_label', type=int, default=1, required=True, help='the start index of features in dataset(if no labels in dataset, this is 1; else this is 2)')
#parser.add_argument('--fun_list', nargs='+', type=str, required=True, help='the fun list of plotting')
parser.add_argument('--fun_type', type=str, required=False, choices = ["plot_distribution", "plot_ks", "plot_odds", "plot_gini", "plot_divergence"], help='the fun type you need assign')
parser.add_argument('--bin_num', type=int, default=100, help='the number of bin')
parser.add_argument('--y_bad_value', type=int, default=1, help='the bad value of y in the dataset')
parser.add_argument('--ratio', type=float, default=0.05, help='the ratio of removing outlier in the plot of distribution')
parser.add_argument('--fig_save_path', type=str, default='fig/',help='the path of saving the figs')
parser.add_argument('--feature_file', type=str, help='the file of feature list')
parser.add_argument('--feature_name', type=str, help= 'the feature name ')
args = parser.parse_args()


def norm(feature):
    #print "norm() begin..."
    #print "len(feature): "+str(len(feature))
    feature_max_value = np.max(feature)
    feature_min_value = np.min(feature)
    feature_after = np.ones(len(feature))
    #print "len(feature_after): "+str(len(feature_after))
    for i in range(len(feature)):
        feature_after[i] = ((feature[i] - feature_min_value) / (feature_max_value - feature_min_value)) * 100
    return feature_after, feature_min_value, feature_max_value

# 去掉头部和尾部的点
def remove_top_bottom(feature, y, ratio = 0.05, has_label=2):
    #print max(feature), min(feature)
    feature_sorted = sorted(feature)
    #print feature_sorted
    feature_sorted_index = np.argsort(feature)
    range_left = int(ratio * len(feature))
    range_right = int((1-ratio) * len(feature))
    feature_after = feature_sorted[range_left:range_right]
    if y is None:
        return feature_after, y
    y = np.array(y)
    y_sorted = y[feature_sorted_index]
    y_after = y_sorted[range_left:range_right]
    return feature_after, y_after

def read_file_all(file_name,feature_name,has_label=2):
    x = []
    if has_label== 1:
        y = None
    else:
        y = []

    with open(file_name) as fin:
        for line in fin.readlines():
            tokens = line.strip().split('\t')
            feature_str = tokens[has_label]
            feature_tokens = feature_str.split(',')
            feature = []
            for ft in feature_tokens:
                ft_tokens = ft.split(':')
                one_feature_name = ft_tokens[0]
                if one_feature_name == feature_name:
                    x.append(float(ft_tokens[1]))
            if has_label == 2:
                label = int(tokens[has_label-1])
                y.append(label)
    return np.array(x), y

# 画KS图
def plot_ks_sub(bad_ks_result_list, good_ks_result_list, index_list, max_ks_gap_index, fig_save_path, feature_name, max_ks_gap_left,max_ks_gap_right, xtick_label_list, title_name, KS_type):
    fig = plt.figure(figsize=(10,10))
    axes = fig.gca()
    axes.plot(index_list,bad_ks_result_list,'r',linewidth=2,label='bad')
    axes.plot(index_list,good_ks_result_list,'g',linewidth=2,label='good')
    axes.legend()
    axes.set_title(title_name + ' ' + KS_type)

    axes_step = int(len(xtick_label_list) / 10)
    xticks = [i*axes_step for i in range(int(len(xtick_label_list)/axes_step+1))]
    axes.set_xticks(xticks)

    xtick_label_list_after = [xtick_label_list[i]for i in range(len(xtick_label_list)) if i % axes_step == 0]
    axes.set_xticklabels(xtick_label_list_after)

    max_ks_gap_good_value = good_ks_result_list[max_ks_gap_index]
    max_ks_gap_bad_value = bad_ks_result_list[max_ks_gap_index]
    annotate_text_y_index = abs(max_ks_gap_bad_value - max_ks_gap_good_value) / 2 + \
                            min(max_ks_gap_good_value, max_ks_gap_bad_value)
    max_ks_gap_value = abs(max_ks_gap_bad_value - max_ks_gap_good_value)
    axes.annotate(str(round(max_ks_gap_value, 2)),xy=(max_ks_gap_index,annotate_text_y_index))
    xytext_value = str(max_ks_gap_index) + ':' + '(' + str(max_ks_gap_left) + ',' + str(max_ks_gap_right) +')'
    axes.annotate(xytext_value, xy=(max_ks_gap_index,0),xytext=(max_ks_gap_index,0.05),arrowprops=dict(facecolor='red',shrink=0.05)) #, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    #axes.annotate('K-S',xy=(max_ks_gap_index,good_ks_result_list[max_ks_gap_index]))
    axes.plot([max_ks_gap_index,max_ks_gap_index],[bad_ks_result_list[max_ks_gap_index],good_ks_result_list[max_ks_gap_index]],linestyle='--',linewidth=2.5)

    # print str(fig_save_path)+str(feature_name)
    fig.savefig(str(fig_save_path)+str(feature_name)+'_'+KS_type.lower()+'.png',dpi=180)
    plt.close(fig)
    #plt.show()

# 画离散度图像
def plot_divergence_sub(bad_divergence_result_list, good_divergence_result_list, index_list, fig_save_path, feature_name, title_name):
    fig = plt.figure(figsize=(10,10))
    axes = fig.gca()
    axes.plot(index_list, bad_divergence_result_list,label='bad')
    axes.plot(index_list, good_divergence_result_list,label ='good')
    axes.legend()
    axes.set_title(title_name + ' Divergence')
    fig.savefig(fig_save_path + feature_name + '_divergence.png',dpi=180)
    plt.close(fig)

# 画Odds图
def plot_odds_sub(odds_result_list, index_list, bad_equal_good_index_list, fig_save_path, feature_name, title_name):
    fig = plt.figure(figsize = (10,10))
    axes = fig.gca()
    axes.plot(index_list,odds_result_list)
    axes.set_title(title_name + ' Odds')
    fig.savefig(fig_save_path + feature_name + '_odds.png',dpi=180)
    plt.close(fig)

# 画jini图
def plot_gini_sub(bad_gini_result_list, index_list, fig_save_path, feature_name, title_name):
    fig = plt.figure(figsize = (10,10))
    axes = fig.gca()
    axes.plot(index_list, bad_gini_result_list)
    axes.set_title(title_name + ' Gini')
    fig.savefig(fig_save_path + feature_name + '_gini.png',dpi=180)
    #plt.show()
    plt.close(fig)

# 画离散度图像
def plot_divergence(feature, y, bin_num, fig_save_path, y_bad_value, feature_name, title_name):
    feature_max_value = np.max(feature)
    feature_min_value = np.min(feature)
    if feature_max_value == feature_min_value:
        print "===This feature ("+ feature_name +  ") has the same value, no figure is draw==="
        return
    feature, feature_min_value_ori, feature_max_value_ori = norm(feature) #normalize the feature [0,100]
    feature_max_value = np.max(feature)
    feature_min_value = np.min(feature)
    bin_value = (feature_max_value - feature_min_value) / bin_num
    bin_list = [(0,0) for i in range(bin_num)]
    bad_all_num = 0
    good_all_num = 0
    for i in range(len(feature)):
        feature_value = feature[i]
        bin_index = int((feature_value - feature_min_value) / bin_value) # the bin section is ( ]
        if bin_index == bin_num:
            bin_index = bin_num -1
        bin_bad_num, bin_good_num = bin_list[bin_index]
        label = y[i]
        if y[i] == y_bad_value:
            bin_bad_num += 1
            bad_all_num += 1
        else:
            bin_good_num += 1
            good_all_num += 1

        bin_list[bin_index] = (bin_bad_num, bin_good_num)
    # bad_accumulate_num = 0
    # good_accumulate_num = 0
    bad_divergence_result_list = []
    good_divergence_result_list = []
    index_list = [i for i in range(bin_num)]
    for bin_bad_num, bin_good_num in bin_list:
        # bad_accumulate_num += bin_bad_num
        # good_accumulate_num += bin_good_num
        bad_divergence_result_list.append(bin_bad_num / len(feature))
        good_divergence_result_list.append(bin_good_num / len(feature))
    plot_divergence_sub(bad_divergence_result_list, good_divergence_result_list, index_list, fig_save_path, feature_name, title_name)

# 画gini图像
def plot_gini(feature, y, bin_num, fig_save_path, y_bad_value, feature_name, title_name):
    feature_sorted = sorted(feature)
    feature_sorted_index = np.argsort(feature)
    bin_list = [0 for i in range(bin_num)]
    bin_range = len(feature_sorted) / bin_num
    bad_all_num = 0

    for i in range(len(feature_sorted)):
        bin_index = int(i / bin_range)
        # print bin_index
        if bin_index >= bin_num:
            bin_index = bin_num-1
        if y[feature_sorted_index[i]] == y_bad_value:
            bad_all_num += 1
            bin_list[bin_index] += 1
    bad_accumulate_number = 0
    bad_gini_result_list = []

    bad_gini_result_list.append(0)
    index_list = [i for i in range(bin_num+1)]
    for i in range(bin_num):
        bad_accumulate_number += bin_list[i]
        bad_gini_result_list.append(bad_accumulate_number / bad_all_num)
    plot_gini_sub(bad_gini_result_list, index_list, fig_save_path, feature_name, title_name)

#从排序数据中提取bin list (史岩的算法)  针对离散值,每个值作为一个bin 对于连续值
def collect_bin_list(x, y, cur_bin_num, bin_num, bin_list, y_bad_value, bin_label_list):
    if len(x)==0 or cur_bin_num == bin_num:
        return
    #每个bin的序号
    index = int(len(x) / (bin_num - cur_bin_num))

    if index >= len(x):
        index = len(x)-1

    real_index = -1
    bin_bad_num = 0
    bin_good_num = 0
    # print cur_bin_num, len(x), index
    for i in range(len(x)):
        if x[i] <= x[index]:
            real_index = i
        else:
            break
        if y_bad_value == y[i]:
            bin_bad_num += 1
        else:
            bin_good_num += 1
    bin_list[cur_bin_num] = (bin_bad_num, bin_good_num)
    bin_label_list[cur_bin_num] = x[index]

    #print cur_bin_num, real_index

    collect_bin_list(x[real_index+1:], y[real_index+1:], cur_bin_num+1, bin_num, bin_list, y_bad_value, bin_label_list)

# 画KS图像
def plot_ks(feature, y, bin_num, fig_save_path, y_bad_value, feature_name, title_name=''):
    #print "plot_ks() begin..."
    plot_ks_population(feature, y, bin_num, fig_save_path, y_bad_value, feature_name, title_name)
    maxvalue = plot_ks_score(feature, y, bin_num, fig_save_path, y_bad_value, feature_name, title_name)
    return maxvalue

# 根据人口数画KS图
def plot_ks_population(feature, y, bin_num, fig_save_path, y_bad_value, feature_name, title_name=''):
    feature_sorted = sorted(feature)
    feature_sorted_index = np.argsort(feature)
    y = np.array(y)
    y_sorted = y[feature_sorted_index]
    bin_list = [(0,0) for i in range(bin_num)]
    bin_label_list = [feature_sorted[-1] for i in range(bin_num)]
    # print bin_label_list
    collect_bin_list(feature_sorted, y_sorted, 0, bin_num, bin_list, y_bad_value, bin_label_list)
    bad_all_num = 0
    good_all_num = 0

    for bin_bad_num, bin_good_num in bin_list:
        bad_all_num += bin_bad_num
        good_all_num += bin_good_num

    bad_accumulate_num = 0
    good_accumulate_num = 0
    bad_ks_result_list = []
    good_ks_result_list = []
    max_ks_gap = -np.inf
    max_ks_gap_index = -1
    bad_ks_result_list.append(0)
    good_ks_result_list.append(0)
    index_list = [i for i in range(bin_num+1)]
    print bin_list
    for i, (bin_bad_num, bin_good_num) in enumerate(bin_list):
        bad_accumulate_num += bin_bad_num
        good_accumulate_num += bin_good_num
        print good_accumulate_num
        bad_ks_result = bad_accumulate_num / bad_all_num
        good_ks_result = good_accumulate_num / good_all_num
        bad_ks_result_list.append(bad_ks_result)
        good_ks_result_list.append(good_ks_result)
        good_bad_ks_gap =  abs(bad_ks_result - good_ks_result)
        if good_bad_ks_gap > max_ks_gap:
            max_ks_gap = good_bad_ks_gap
            max_ks_gap_index = i

    # bin_label_list = [feature_sorted[int(i * bin_range)] for i in range(bin_num)]
    bin_label_list.append(max(feature_sorted))


    max_ks_gap_left = bin_label_list[max_ks_gap_index]
    max_ks_gap_right = bin_label_list[max_ks_gap_index+1]

    xtick_label_list = [str(0) + '\n' + str(0)]
    for i in index_list[1:]:
        xtick_label_list.append(str(i)+'\n'+str(round(bin_label_list[i],2)))

    max_ks_gap_index += 1

    KS_type = 'KS_by_Population'
    plot_ks_sub(bad_ks_result_list,good_ks_result_list,index_list,max_ks_gap_index,fig_save_path,feature_name,max_ks_gap_left,max_ks_gap_right, xtick_label_list,title_name,KS_type)

# 根据Score画KS图像
def plot_ks_score(feature, y, bin_num, fig_save_path, y_bad_value, feature_name, title_name=''):
    #print "plot_ks_score() begin..."
    feature_max_value = np.max(feature)
    feature_min_value = np.min(feature)
    #print feature_min_value, feature_max_value
    if feature_max_value == feature_min_value:
        # print "===This feature (" + feature_name +  ") has the same value, no figure is draw==="
        print "this feature  has the same value, no figure is draw==="
        return

    feature, feature_min_value_ori, feature_max_value_ori = norm(feature) #normalize the feature [0,100]
    feature_max_value = np.max(feature)
    feature_min_value = np.min(feature)
    #print feature_min_value_ori, feature_max_value_ori
    #print feature_min_value, feature_max_value
    bin_value = (feature_max_value - feature_min_value) / bin_num
    bin_list = [(0,0) for i in range(bin_num)]
    bad_all_num = 0
    good_all_num = 0
    for i in range(len(feature)):
        feature_value = feature[i]
        bin_index = int((feature_value - feature_min_value) / bin_value)
        if bin_index == bin_num:
            bin_index = bin_num -1
        bin_bad_num, bin_good_num = bin_list[bin_index]
        label = y[i]
        if y[i] == y_bad_value:
            bin_bad_num += 1
            bad_all_num += 1
        else:
            bin_good_num += 1
            good_all_num += 1

        bin_list[bin_index] = (bin_bad_num, bin_good_num)
    bad_accumulate_num = 0
    good_accumulate_num = 0
    bad_ks_result_list = []
    good_ks_result_list = []
    max_ks_gap = np.inf
    max_ks_gap_index = -1
    bad_ks_result_list.append(0)
    good_ks_result_list.append(0)
    index_list = [i for i in range(bin_num+1)]
    for i, (bin_bad_num, bin_good_num) in enumerate(bin_list):
        bad_accumulate_num += bin_bad_num
        good_accumulate_num += bin_good_num
        bad_ks_result = bad_accumulate_num / bad_all_num
        good_ks_result = good_accumulate_num / good_all_num
        bad_ks_result_list.append(bad_ks_result)
        good_ks_result_list.append(good_ks_result)
        bad_good_ks_gap = abs(bad_ks_result - good_ks_result)
        if bad_good_ks_gap > max_ks_gap:
            max_ks_gap = bad_good_ks_gap
            max_ks_gap_index = i

    # print "max_ks_gap: " + str(max_ks_gap)
    max_ks_gap_left = max_ks_gap_index * bin_value + feature_min_value
    max_ks_gap_right = (max_ks_gap_index+1) * bin_value + feature_min_value
    max_ks_gap_left = max_ks_gap_left / 100 * (feature_max_value_ori - feature_min_value_ori) + feature_min_value_ori
    max_ks_gap_right = max_ks_gap_right / 100 * (feature_max_value_ori - feature_min_value_ori) + feature_min_value_ori
    # print max_ks_gap_left,max_ks_gap_right
    max_ks_gap_index += 1
    xtick_label_list = [str(0) + '\n' + str(0)]
    for i in index_list[1:]:
        bottom_value=((i * bin_value + feature_min_value) * (feature_max_value_ori - feature_min_value_ori) / 100 )
        xtick_label_list.append(str(i)+'\n'+str(round(bottom_value, 2))) # 最右端
        #xtick_label_list.append(str(i)+'\n'+str((i * bin_value + feature_min_value)/100 * (feature_max_value_ori - feature_min_value_ori) # 最右端
        #                + feature_min_value_ori))

    #print feature_min_value_ori, feature_max_value_ori
    KS_type = 'KS_By_Score'
    plot_ks_sub(bad_ks_result_list,good_ks_result_list,index_list,max_ks_gap_index,fig_save_path,feature_name,max_ks_gap_left,max_ks_gap_right, xtick_label_list,title_name, KS_type)
    return max_ks_gap
# 画Odds图像
def plot_odds(feature, y, bin_num, fig_save_path, y_bad_value,feature_name,title_name=''):
    feature_max_value = np.max(feature)
    feature_min_value = np.min(feature)
    if feature_max_value == feature_min_value:
        print "===This feature ("+ feature_name +  ") has the same value, no figure is draw==="
        return
    feature, feature_min_value_ori, feature_max_value_ori = norm(feature) #normalize the feature [0,100]
    feature_max_value = np.max(feature)
    feature_min_value = np.min(feature)
    bin_value = (feature_max_value - feature_min_value) / bin_num
    bin_list = [(0,0) for i in range(bin_num)]
    bad_all_num = 0
    good_all_num = 0
    for i in range(len(feature)):
        feature_value = feature[i]
        bin_index = int((feature_value - feature_min_value) / bin_value) # the bin section is ( ]
        if bin_index == bin_num:
            bin_index = bin_num -1
        bin_bad_num, bin_good_num = bin_list[bin_index]
        label = y[i]
        if y[i] == y_bad_value:
            bin_bad_num += 1
            bad_all_num += 1
        else:
            bin_good_num += 1
            good_all_num += 1

        bin_list[bin_index] = (bin_bad_num, bin_good_num)
    index_list = [i for i in range(bin_num)]
    # bad_odds_result_list = []
    # good_odds_result_list = []
    odds_result_list = []
    bad_equal_good_index_list = []
    for i, (bin_bad_num, bin_good_num) in enumerate(bin_list):
        bad_odds_result = bin_bad_num / bad_all_num
        #bad_odds_result_list.append(bad_odds_result)
        good_odds_result = bin_good_num / good_all_num
        # print bad_odds_result,good_odds_result
        if good_odds_result == 0:
            odds_result_list.append(np.log(10000/3))
        elif bad_odds_result == 0:
            odds_result_list.append(np.log(3/10000))
        else:
            odds_result_list.append(np.log(bad_odds_result / good_odds_result) ) #分母为0？
        #good_odds_result_list.append(good_odds_result)
        if good_odds_result == bad_odds_result:
            bad_equal_good_index_list.append(i)
    plot_odds_sub(odds_result_list, index_list, bad_equal_good_index_list,fig_save_path,feature_name, title_name)

# 画分布图
def plot_distribution_label(feature, y, bin_num, fig_save_path, y_bad_value, feature_name = '', title_name='', ratio = 0.05):
    feature = np.array(feature)
    y = np.array(y)
    feature_max_value = np.max(feature)
    feature_min_value = np.min(feature)
    # print feature_max_value, feature_min_value
    bin_value = (feature_max_value - feature_min_value) / bin_num
    if bin_value == 0:
        print "===This feature ("+ feature_name +  ") is all zero, no figure is draw==="
        return
    bin_list = [(0,0) for i in range(bin_num)] #在bin内的点数 在bin内的good 在bin内的bad
    for i, v in enumerate(feature):
        bin_index = int((v - feature_min_value) / bin_value)
        if bin_index >= bin_num:
            bin_index = bin_num - 1
        good_num, bad_num = bin_list[bin_index]
        if y[i] == y_bad_value:
            bad_num += 1
        else:
            good_num += 1
        bin_list[bin_index] = (good_num, bad_num)

    bin_index = [i for i in range(bin_num)]
    xticks_index = [feature_min_value+i*bin_value for i in bin_index]
    bad_num_list = []
    good_num_list = []
    for good_num, bad_num in bin_list:
        good_num_list.append(good_num)
        bad_num_list.append(bad_num)

    # 绘制distribution_label
    fig1 = plt.figure(figsize = (10,10))
    axes1 = fig1.gca()
    bar_width = 0.5
    axes_step = int(len(bin_index) / 20)
    xticks = [i*axes_step + bar_width/2 for i in range(int(len(bin_index)/axes_step+1))]
    axes1.set_xticks(xticks)
    xtick_label_list_after = [xticks_index[i]for i in range(len(xticks_index)) if i % axes_step == 0]

    axes1.bar(bin_index, good_num_list, bar_width, color='b', label = 'good')
    axes1.bar(bin_index, bad_num_list, bar_width, color = 'r', label = 'bad', bottom = good_num_list)
    #axes1.get_xaxis().set_major_locator(LinearLocator(numticks=11))
    axes1.set_xticks([i*axes_step + bar_width/2 for i in range(int(len(bin_index)/axes_step+1))])
    axes1.set_xticklabels(xtick_label_list_after, rotation=90)
    axes1.set_title(title_name + ' distribution label')
    axes1.legend()
    fig1.savefig(fig_save_path + feature_name + '_distribution_label.png', dpi=180)
    plt.close(fig1)

    # 绘制norm后的distribution_label
    for i in range(len(good_num_list)):
        good_num = good_num_list[i]
        bad_num = bad_num_list[i]
        sum_num = good_num + bad_num
        if good_num + bad_num != 0:
            good_num_list[i] = good_num / sum_num
            bad_num_list[i] = bad_num / sum_num
    fig2 = plt.figure(figsize=(10,10))
    axes2 = fig2.gca()
    bar_width = 0.5
    axes2.bar(bin_index,good_num_list,bar_width,color='b',label='good')
    axes2.bar(bin_index,bad_num_list,bar_width,color='r',label='bad',bottom=good_num_list)
    axes2.set_xticks([i*axes_step + bar_width/2 for i in range(int(len(bin_index)/axes_step+1))])
    axes2.set_xticklabels(xtick_label_list_after,rotation=90)
    axes2.set_title(title_name + ' distributiob label norm')
    axes2.legend()
    axes2.set_ylim(0,1.2)
    fig2.savefig(fig_save_path + feature_name + '_distribution_label_norm.png', dpi=180)
    plt.close(fig2)

def plot_distribution_num(feature, bin_num, fig_save_path, y_bad_value, feature_name = '', title_name='', ratio = 0.05):
    feature = np.array(feature)
    feature_max_value = np.max(feature)
    feature_min_value = np.min(feature)
    #print feature_max_value, feature_min_value
    bin_value = (feature_max_value - feature_min_value) / bin_num
    if bin_value == 0:
        print "This feature is all zero"
        return
    bin_list = [0 for i in range(bin_num)] #在bin内的点数
    for i, v in enumerate(feature):
        bin_index = int((v - feature_min_value) / bin_value)
        if bin_index >= bin_num:
            bin_index = bin_num - 1
        bin_list[bin_index] += 1

    bin_index = [i for i in range(bin_num)]
    xticks_index = [feature_min_value+i*bin_value for i in bin_index]
    num_list = []

    for num in bin_list:
        num_list.append(num)
    fig = plt.figure("1",figsize = (10,10))
    axes = fig.gca()
    bar_width = 0.5

    axes.bar(bin_index, num_list, bar_width, color='r')
    axes.set_title(title_name + ' distribution')
    axes_step = int(len(num_list) / 20)
    xticks = [i*axes_step + bar_width/2 for i in range(int(len(num_list)/axes_step+1))]
    axes.set_xticks(xticks)
    xtick_label_list_after = [xticks_index[i]for i in range(len(xticks_index)) if i % axes_step == 0]
    axes.set_xticklabels(xtick_label_list_after, rotation=90)
    fig.savefig(fig_save_path + feature_name + '_distribution_num.png', dpi=180)
    plt.close(fig)

# 画分布图
def plot_distribution(feature, y, bin_num, fig_save_path, y_bad_value, feature_name = '', title_name='', ratio = 0.05, has_label=2):
    feature, y = remove_top_bottom(feature, y, ratio, has_label)
    if has_label==1: # 表示没有标签
        plot_distribution_num(feature, bin_num, fig_save_path, y_bad_value, feature_name, title_name, ratio)
    elif has_label==2: # 表示有标签
        plot_distribution_num(feature, bin_num, fig_save_path, y_bad_value, feature_name, title_name, ratio)
        plot_distribution_label(feature, y, bin_num, fig_save_path, y_bad_value, feature_name, title_name, ratio)

# 对一个特征进行画图
def plot_one_feature(feature, y, bin_num, fig_save_path, fun_name_list, y_bad_value=1, feature_name = '', title_name='', dist_ratio=0.05):
    feature_name = np.array(feature_name)
    maxvalue = plot_ks(feature, y, bin_num, fig_save_path, y_bad_value, feature_name, title_name='')
    # for fun_name in fun_name_list:
    #     maxvalue = eval(fun_name)(feature, y, bin_num, fig_save_path, y_bad_value, feature_name, title_name)
    return maxvalue
# 画出一个特征的所有图像
def plot_one_feature_all(file_name, feature_name, bin_num, fig_save_path, fun_list, y_bad_value=1, title_name='', has_label=2, dist_ratio=0.05):
    x, y = read_file_all(file_name, feature_name, has_label)
    check_y(y, y_bad_value)
    for fun_name in fun_list:
        print "func_name: "+fun_name
        if has_label == 1 and fun_name != 'plot_distribution':
            raise ValueError('=============Cannot plot other figures as no label============')
        if has_label==2 and fun_name=="plot_ks":
            plot_ks(x, y, bin_num, fig_save_path, y_bad_value, feature_name, title_name)
        if fun_name == 'plot_distribution':
            eval(fun_name)(x, y, bin_num, fig_save_path, y_bad_value, feature_name, title_name, dist_ratio, has_label)
        else:
            eval(fun_name)(x, y, bin_num, fig_save_path, y_bad_value, feature_name, title_name)

# 从指定文件中读取数据并且画图
def plot_one_feature_file(file_x,file_y,bin_num,fig_save_path,fun_name_list,y_bad_value=1,feature_name='',title_name=''):
    x = []
    y = []
    with open(file_x, 'rb') as f_x:
        for line in f_x.readlines():
            tokens = line.strip().split('\t')
            x.append(float(tokens[0]))
    with open(file_y, 'rb') as f_y:
        for line in f_y.readlines():
            tokens = line.strip().split('\t')
            y.append(float(tokens[0]))
    x = np.array(x)
    y = np.array(y)
    for fun_name in fun_name_list:
        eval(fun_name)(x, y,bin_num,fig_save_path,y_bad_value,feature_name,title_name)

# 从LR训练结果文件中读取数据并且画图
def plot_one_feature_lr(file_x,file_y,bin_num,fig_save_path,fun_name_list,y_bad_value=1,feature_name='',title_name=''):
    #print "plot_one_feature_lr() begin..."
    x = []
    y = []
    file_x_lines = open(file_x, 'rb').readlines()
    for i in range(1,len(file_x_lines)):
        tokens = file_x_lines[i].strip().split(' ')
        x.append(float(tokens[1]))
    file_y_lines = open(file_y, 'rb').readlines()
    for i in range(0,len(file_y_lines)):
        tokens = file_y_lines[i].strip().split(' ')
        y.append(float(tokens[0]))
    if len(x)!=len(y):
        print "x_len_not_equal_y!"
        exit()
    x = np.array(x)
    y = np.array(y)
    for fun_name in fun_name_list:
        print "fun_name: "+fun_name
        eval(fun_name)(x, y,bin_num,fig_save_path,y_bad_value,feature_name,title_name)

# 从GBDT训练结果文件中读取数据并且画图
def plot_one_feature_gbdt(file_x,file_y,bin_num,fig_save_path,fun_name_list,y_bad_value=1,feature_name='',title_name=''):
    #print "plot_one_feature_gbdt() begin..."
    x = []
    y = []
    file_x_lines = open(file_x, 'rb').readlines()
    for i in range(0,len(file_x_lines)):
        tokens = file_x_lines[i].strip().split('\t')
        x.append(float(tokens[0]))
    file_y_lines = open(file_y, 'rb').readlines()
    for i in range(0,len(file_y_lines)):
        tokens = file_y_lines[i].strip().split(' ')
        y.append(float(tokens[0]))
    if len(x)!=len(y):
        print "x_len_not_equal_y!"
        exit()
    x = np.array(x)
    y = np.array(y)
    for fun_name in fun_name_list:
        print "fun_name: "+fun_name
        eval(fun_name)(x, y,bin_num,fig_save_path,y_bad_value,feature_name,title_name)

# 从输入文件中读取所有特征的名称
def gen_feature_names_from_data(data_path, has_label=2):
    with open(data_path,'rb') as f:
        feature_name_list = []
        for line in f.readlines():
            if line == "":
                continue
                # print "line_is_empty"
                # exit()
            tokens = line.strip().split('\t')
            for feature in tokens[has_label].split(','):
                f_tokens = feature.split(':')
                feature_name_list.append(f_tokens[0])
            return feature_name_list

# 从一个指定文件中获取特征名称
def gen_feature_name_from_file(file):
    with open(file) as f:
        feature_name_list = []
        for line in f.readlines():
            if line == "":
                continue
            feature_name_list.append(line.strip())
    return feature_name_list

# 检查label
def check_y(y, y_bad_value):
    good_all = 0
    bad_all = 0
    for i in y:
        if i == y_bad_value:
            bad_all += 1
        else:
            good_all += 1
    if good_all == 0 or bad_all== 0:
        raise ValueError("================Only one label in y, please check your data=================")

# 检查参数
def check_arguments(data_path, fig_save_path, feature_file=None):
    """检查data_path是否存在，如果不存在抛出错误; 检查fig_save_path 是否存在，如果不存在创建
    """
    if not os.path.exists(data_path):
        raise NameError("================The input data path dose not exist==============")

    if not os.path.exists(fig_save_path):
        print "The path of saving figs dose not exits. "
        os.makedirs(fig_save_path)
        print "The path of saving figs has been created"

    if fig_save_path[-1] != '/':
        fig_save_path += '/'

    if feature_file:
        if not os.path.exists(feature_file):
            raise NameError("===The input feature list dose not exist==============")
    return data_path, fig_save_path

if __name__ == '__main__':
    data_path = args.data_path
    has_label = args.has_label
    #has_label = args.has_label
    fun_type = args.fun_type
    fun_list = []
    if fun_type==None:
        fun_list = ["plot_distribution", "plot_ks", "plot_odds", "plot_gini", "plot_divergence"]
    else:
        fun_list.append(fun_type)
    bin_num = args.bin_num
    y_bad_value = args.y_bad_value
    ratio = args.ratio
    fig_save_path = args.fig_save_path
    feature_file = args.feature_file
    feature_name = args.feature_name
    feature_name_list = []

    data_path, fig_save_path = check_arguments(data_path, fig_save_path, feature_file)
    if feature_file is None and feature_name is None:
        print "===feature file and feature name are not assigned, so all features will be draw==="
        feature_name_list = gen_feature_names_from_data(data_path, has_label)
        #feature_name_list = gen_feature_names_from_data(data_path, has_label)
    elif feature_file and feature_name:
        raise NameError("========Cannot assign feature file and feature name both==========")
    elif feature_file:
        feature_name_list = gen_feature_name_from_file(feature_file)
    else:
        feature_name_list.append(feature_name)
    #print feature_file

    for feature_name in feature_name_list:
        print "feature_name: "+feature_name
        #print data_path,feature_name,bin_num,fig_save_path,y_bad_value,feature_name,has_label
        plot_one_feature_all(data_path, feature_name, bin_num, fig_save_path, fun_list, y_bad_value, feature_name, has_label)


