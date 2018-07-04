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

parser = argparse.ArgumentParser()
parser.add_argument('--data_name',type=str,default='data_all/xxd_version1',help='training data in csv format')
parser.add_argument('--term',type=str,default=24,help='training data in csv format')
parser.add_argument('--gap_name',type=str,default='data_all/xxd_version1_gap_24.csv',help='training data in csv format')
parser.add_argument('--save_name',type=str,default='data_all/xxd_version1_des_24.csv',help='training data in csv format')
args = parser.parse_args()

"""
A:true ok   predict ok
B:true ok   predict M7
C:true M7   predict ok
DB:true M7   predict M7
"""

def load_data(path):
    return pd.read_csv(path, error_bad_lines=False)


def outliers_detection(data, times = 7, quantile = 0.95):
    data=data[-data.isnull()]
    data = np.array(sorted(data))
    #std-outlier
    outlier1 = np.mean(data) + 1*np.std(data)

    # mad-outlier
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    outlier2 = med + times * mad

    # quantile-outlier
    outlier3 = data[int(np.floor(quantile * len(data)) - 1)]
    return outlier1, outlier2, outlier3



#画离连续图
def good_ratio_plot(data, col_name,SP,fig_save_path,m=2000):
    data_notnull = data[-data[col_name].isnull()]
    if len(data_notnull)<m:
        pass
    else:
        #######################  数据排序  #######################
        index = np.argsort(data_notnull[col_name])
        sorted_col=list(pd.Series(list(data_notnull[col_name]))[index])#排序后的列值
        sorted_label=list(pd.Series(list(data_notnull['label']))[index])#排序后的标签
        n=len(sorted_col)

        ###################  排序后, 把值相同的归为一组 ###################
        index2=[0]#the starting location for each group
        i=0
        while i<n-1:
            if(sorted_col[i]==sorted_col[i+1]):
                i=i+1
            else:
                index2.append(i+1)
                i=i+1

        num_group = len(index2)
        print num_group
        #把值相同的归为一组
        group_data=[sorted_col[index2[i]: index2[i + 1]] for i in range(0, num_group - 1)]
        if(index2[-1]==n-1):
            group_data.append([sorted_col[-1]])
        else:
            group_data.append(sorted_col[index2[-1]:n])

        # 按照值的分组把标签分组
        group_label=[sorted_label[index2[i]: index2[i + 1]] for i in range(0, num_group - 1)]
        if(index2[-1]==n-1):
            group_label.append([sorted_label[-1]])
        else:
            group_label.append(sorted_label[index2[-1]:n])

        ##################  计算累计百分比  ###################\
        len_list=[]
        sum_list=[]
        N = []  # 存储累计百分比
        cur_sum = 0
        total_count = len(sorted_col)
        for s in range(0, num_group):
            len_list.append(len(group_data[s]))
            sum_list.append(sum(group_label[s]))
            cur_sum = cur_sum + len(group_data[s])
            N.append(cur_sum * 1.0 / total_count)

        ###################  计算好人比例与密度 ###################
        ratio=[]#存储好人比例
        #计算开头 及其比例
        density=[]#存储密度
        cur_cnt=len_list[0]
        for first_m in range(0,num_group):
            if(cur_cnt<m):
                cur_cnt=cur_cnt+len_list[first_m+1]
            else:
                break

        # print 'first_m:',first_m

        if(first_m%2==1):
            first_point=(first_m-1)/2
        if(first_m%2==0):
            first_point=first_m/2
        # print 'first_point:',first_point


        total_cnt1 = len_list[0:first_m+1]
        total_good1 = sum_list[0:first_m+1]
        ratio_1=sum(total_good1)*1.0/sum(total_cnt1)
        for i in range(0,first_point+1):
            ratio.append(ratio_1)

        if(first_m%2==1 and first_m+1<=num_group-1):
            ratio.append((sum(total_good1) + sum_list[first_m + 1]) * 1.0 / (sum(total_cnt1) + len_list[first_m + 1]))
            first_point=first_point+1
            first_m=first_m+1

        # print 'first ratio:', ratio


        forward_pointer = first_m
        backward_pointer = 0


        for i in range(first_point+1,num_group):
            #print 'i:',i
            #判断单个是否大于m,如果单个大于m则直接化为单个bin
            if(len_list[i]>=m):
                ratio.append(sum_list[i]*1.0/len_list[i])
                forward_pointer=i
                backward_pointer=i
            else:
                #到结尾的时候,将之后的数据延伸为一个bin
                if(forward_pointer==num_group -1):
                    total_cnt_end = len_list[backward_pointer: forward_pointer + 1]  # 把延伸后的数据合成一个组
                    total_good_end = sum_list[backward_pointer: forward_pointer + 1]
                    ratio_end = sum(total_good_end) * 1.0 / sum(total_cnt_end)
                    for j in range(i, forward_pointer + 1):
                        ratio.append(ratio_end)
                    print backward_pointer,forward_pointer
                    break

                else:
                    # 在中间的部分,并且没到最后一个bin则执行前后相比原则。
                    if(backward_pointer!=forward_pointer):
                        #前后相比原则 or 虽然后面比前面小但是仍大于m 仍然是向右移动一个位置
                        if (len_list[forward_pointer + 1] > len_list[backward_pointer] or sum(len_list[backward_pointer + 1: forward_pointer + 2]) >= m):
                            forward_pointer = forward_pointer + 1
                            backward_pointer = backward_pointer+1
                        else:
                            #否则向右移动两个位置
                            forward_pointer = forward_pointer + 2 if forward_pointer + 2 <= num_group - 1 else num_group - 1
                    else:
                        #由于单个bin的生成导致指针位置的改变
                        forward_pointer = i + 1 if i + 1 <= num_group - 1 else num_group - 1
                    total_cnt = len_list[backward_pointer: forward_pointer + 1]  # 把延伸后的数据合成一个组
                    total_good = sum_list[backward_pointer: forward_pointer + 1]
                    ratio.append(sum(total_good) * 1.0 / sum(total_cnt))  #计算好人比例


        #画出好人比例图
        fig1 = plt.figure(figsize=(10, 10))
        plt.plot(N, ratio, 'b')
        plt.xlabel(str(col_name))
        plt.ylabel('good ratio')
        plt.xlim(0,1)
        y0 = sum(sorted_label)*1.0/len(sorted_label)
        plt.axhline(y=y0, linewidth=0.3, color='k')



        fig1.savefig(str(fig_save_path)+"_"+SP+"_"+str(col_name) + '.png', dpi=180)
        return max(ratio), min(ratio), max(ratio) - min(ratio)


#画离散图
def good_ratio_plot_discrete(dataframe,fig_save_path,SP):
    """
        INPUT:
          dataframe: dataframe format. The first column must be label.
        OUTPUT:
          1)dictionary:
            key is feature name,
            value is a list which contains three parts: bin points(list), logodds(list), logodds for null(a number, if have null values)
          2)odds plot: if the x-axis has -1, that means bin for null values.
    """

    fea_list = dataframe.columns  # feature list, the first one is label
    output = {}


    for k in range(1, len(fea_list)):
        col_name = fea_list[k]
        data = dataframe[[col_name, 'label']]
        data_notnull = data[-data[col_name].isnull()]
        col = list(data_notnull[col_name])
        distinct_val = set(col)
        sorted_col = sorted(distinct_val)
        ratio = []
        for val in sorted_col:
            data_tmp = data[data[col_name] == val]
            n_cur = len(data_tmp)
            n_good = len(data_tmp[data_tmp['label'] == 1])
            ratio.append(n_good*1.0/n_cur)

        sorted_col_update=flatten([sorted_col[0],sorted_col[:]])
        sorted_col_update[0]=sorted_col_update[0]-0.5
        sorted_col_update[-1]=sorted_col_update[-1]+0.5
        for i in range(1, len(sorted_col_update) - 1):
            sorted_col_update[i] = (sorted_col_update[i] + sorted_col_update[i+1]) * 1.0 / 2
        output[col_name] = [sorted_col_update, ratio]


        #############################################  logodds plot  #######################################################

        fig = plt.figure(figsize=(10, 10))
        plt.plot(sorted_col, ratio, 'ro')
        plt.plot(sorted_col, ratio, 'b')
        plt.xlabel('values')
        plt.ylabel('good ratio')
        plt.xticks(sorted_col)
        # average level
        #y0 = math.log(sum(data['label'] == 1) * 1.0 / (len(data) - sum(data['label'] == 1)))
        y0=sum(data_notnull['label'] == 1) * 1.0 / len(data_notnull)
        #y0=0.5
        plt.axhline(y=y0, linewidth=0.3, color='k')
        fig.savefig(str(fig_save_path)+"_"+SP+"_"+str(col_name) + '.png', dpi=180)
        # plt.show()
        return max(ratio), min(ratio), max(ratio) - min(ratio)

def spit_term_dataframe(dataframe_my,term):

    start_data = '2014-09-01 00:00:00'
    end_data = '2016-12-01 00:00:00'
    spilt_data='2016-10-01 00:00:00'
    start_data = datetime.datetime.strptime(start_data, '%Y-%m-%d %H:%M:%S')
    end_data = datetime.datetime.strptime(end_data, '%Y-%m-%d %H:%M:%S')

    print len(dataframe_my)

    # dataframe_my=dataframe_my[dataframe_my['loan_term']==term]

    dataframe_my = dataframe_my[dataframe_my["inspection_time"] >= str(start_data)]
    dataframe_my = dataframe_my[dataframe_my["inspection_time"] <str(end_data)]

    traindata=dataframe_my[dataframe_my["inspection_time"] < str(spilt_data)]
    testdata = dataframe_my[dataframe_my["inspection_time"] >= str(spilt_data)]
    print len(dataframe_my),len(traindata),len(testdata)
    return traindata

def Plot_goodradio(new_data,term):
    Continuous = ["a.label","a.label_profit",0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0
                    ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]

    gap_name=args.gap_name
    gap = open(gap_name, 'wb+')
    gap_output = csv.writer(gap, dialect='excel')
    gap_output.writerow(["feature","max","min","gap"])


    fea_list = new_data.columns
    print fea_list

    for m in range(2, len(fea_list)):
        colom = fea_list[m]
        print colom
        print Continuous[m]


        fig_save_path='fig_new_ratio'+str(term)+'/'
        # fig_save_path = str(fig_save_path) + str(colom) + '/'
        if not os.path.exists(fig_save_path):
            os.makedirs(fig_save_path)
            print "The path of saving figs has been created"

        try:
            if Continuous[m]==0:
                print "it is Continuous"
                Max, Min, gap =good_ratio_plot(new_data,colom,str(term),fig_save_path)
            else:
                print "it is not Continuous"
                dataFrame = new_data[['label', colom]]
                Max,Min,gap=good_ratio_plot_discrete(dataFrame,fig_save_path,str(term))
                print Max,Min,gap
            gap_output.writerow([colom,Max,Min,gap])
        except:
            print "err"
def des(data):
    decrib = ['feature', 'count（not nullall_count:761539', 'min', 'max', 'mean', '方差', '偏度', '峰度',
              '25%', '50%', '75%', '90%', '99.7%', '99.97%所占人数', '覆盖率', '划分（针对离散型变量）"	含义	"连续1:离散：0', 'meaning']
    savename=args.save_name
    file0 = open(savename, 'wb+')  # 'wb'
    output = csv.writer(file0, dialect='excel')
    output.writerow(decrib)
    total = len(data)
    fea_list = data.columns

    for m in range(12, len(fea_list)):
        colom = fea_list[m]
        data_notnull = data[-data[colom].isnull()][colom]
        print len(data_notnull)
        g_dist = sorted(data_notnull)
        lenth = len(g_dist)
        info = stats.describe(data_notnull)
        listdes = [colom, str(info[0]), str(info[1][0]), str(info[1][1]), str(info[2]),
                   str(info[3]), str(info[4]), str(info[5]), g_dist[int(0.25 * lenth)],
                   g_dist[int(0.5 * lenth)], g_dist[int(0.75 * lenth)], g_dist[int(0.9 * lenth)],
                   g_dist[int(0.9997 * lenth)], int(lenth - int(0.9997 * lenth)), float(int(info[0]) * 1.0 / total)]
        output.writerow(listdes)


def plot_ratio(data,Label,fig_save_path):
    setcolom = list(set(list(new_data[-new_data[colom].isnull()][colom])))
    dictcolom = {}
    for i in setcolom:
        dictcolom[i] = list(new_data[colom]).count(i)
        print i, ",", dictcolom[i]

    data=data[-data.isnull()]
    print len(data)
    fig = plt.figure(figsize=(10, 10))
    plt.hist( list(data), label=Label, color="b",alpha=0.6,rwidth=0.8)
    plt.legend()
    plt.show()
    fig.savefig(str(fig_save_path) +"_" + str(Label) + '.png', dpi=180)
def desfen(data,colom, ind):
    data_notnull = data[-data[colom].isnull()][colom]
    print len(data_notnull)
    g_dist = sorted(data_notnull)
    lenth = len(g_dist)
    info = stats.describe(data_notnull)

    print g_dist[int(ind * lenth)]
if __name__ == '__main__':

    data_name = args.data_name
    new_data = load_data(data_name)
    term = args.term

    #划分数据集
    new_data = spit_term_dataframe(new_data, term)
    # 对数据集进行基本描述
    # des(new_data)
    print len(new_data)
    #画出数据集的好人比例图
    # Plot_goodradio(new_data,term)


    colom = "APPLY_MAX_AMOUNT"
    #画出分布图
    # plot_ratio(new_data[new_data[colom] < 15][colom], colom, "hist/")

    desfen(new_data,colom, 0.5)


