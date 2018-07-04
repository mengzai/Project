#encoding=utf-8
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from compiler.ast import flatten
import string


#input
data=pd.read_csv('logistic_regression/GENDER_test_24_1')
weight=data['profit']
groud_truth=data['label_profit']
f = open("logistic_regression/test_result.txt","r")
lines = f.readlines()#读取全部内容
i=0
y_pred=[]
for line in lines:
    line=line.split('\t')
    y_pred.append(string.atof(line[2]))
save_path='logistic_regression'


#输入: weight(即profit,list), y_pred(预测概率,list), save_path(存储路径)
def profit_ks_plot(weight, y_pred,save_path):
    weight = pd.Series(weight)
    print 'original net profit = ',sum(weight)
    print 'sample length = ',len(y_pred)
    num_bin = 1000  #number of bins
    y_pred_sorted=pd.Series(sorted(y_pred))
    bin_point=np.array(range(0,num_bin+1))*1.0/num_bin#0,0.01,0.02,.....,1,画图用

    bin_point_loc=np.array(range(0,num_bin+1))*1.0*len(y_pred)/num_bin#找出百分位数的位置
    percent_location=[int(bin_point_loc[i]) for i in range(0,len(bin_point_loc))]#把位置转化成整数
    new_bin_point=list(y_pred_sorted[percent_location[0:num_bin]])#根据位置找出百分位数
    new_bin_point.append(list(y_pred_sorted)[-1])#加上最大值
    pos = np.digitize(y_pred, new_bin_point)#根据百分位数分bin

    bin=pd.Series(pos)
    bin[bin> len(new_bin_point) - 1] = len(new_bin_point) - 1

    group_weight = [list(weight[bin==i]) for i in range(1, len(bin_point))]#把weight分到对应的bin

    #最初全判为坏人,这时好人受益坏人亏损均为0
    gain=[0]
    loss=[0]

    for i in range(0,len(group_weight)):#从最后一个group开始,往前延伸,即逐渐让预测概率大的样本为好人,每次多放过1%
        cur_group=group_weight[(len(group_weight)-1-i):len(group_weight)]
        cur_group=flatten(cur_group)
        cur_group=pd.Series(cur_group)
        gain.append(sum(cur_group[cur_group>=0]))
        loss.append(abs(sum(cur_group[cur_group < 0])))

    diff_list = list(np.array(gain) - np.array(loss))
    max_ks_gap_index = diff_list.index(max(diff_list))
    max_ks_gap_value = max(diff_list)
    print 'max profit = ',max_ks_gap_value
    x = list(bin_point)
    fig = plt.figure(figsize=(10, 10))
    axes = fig.gca()
    axes.plot(x, gain, 'r', linewidth=2, label='gain')
    axes.plot(x, loss, 'g', linewidth=2, label='loss')

    max_ks_gap_bad_value = loss[max_ks_gap_index]
    annotate_text_y_index = max_ks_gap_value/2 + max_ks_gap_bad_value
    xytext_value = str(new_bin_point[len(new_bin_point)-1-max_ks_gap_index])
    print 'threshold = ',new_bin_point[len(new_bin_point)-1-max_ks_gap_index]

    axes.annotate(xytext_value, xy=(max_ks_gap_index * 1.0 / num_bin, 0),
                  xytext=(max_ks_gap_index * 1.0 / num_bin, 0.05),
                  arrowprops=dict(facecolor='red', shrink=0.05))
    axes.plot([max_ks_gap_index * 1.0 / num_bin, max_ks_gap_index * 1.0 / num_bin],
              [loss[max_ks_gap_index], gain[max_ks_gap_index]], linestyle='--',
              linewidth=2.5)
    axes.annotate('Max Profit='+str(int(max_ks_gap_value*1.0/1000000))+'M', xy=(max_ks_gap_index * 1.0 / num_bin, annotate_text_y_index))
    #str(round(max_ks_gap_value, 2))
    axes.legend(loc='upper left')
    plt.xlabel('Accumulated Percentage')
    plt.ylabel('Amount')

    axes.set_title('Profit KS Curve')
    fig.savefig('%s/ks_curve.png' % (save_path), dpi=180)
    plt.show()
    plt.close(fig)

profit_ks_plot(weight,y_pred,save_path)
