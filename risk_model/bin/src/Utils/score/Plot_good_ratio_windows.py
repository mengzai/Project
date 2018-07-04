#-*- coding: UTF-8 -*-
import csv
import argparse
import os
import pandas as pd
import numpy as np
import numpy
import matplotlib.pyplot as plt
from compiler.ast import flatten
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--data_name',type=str,default='data_all/xxd_good_and_m7',help='training data in csv format')
parser.add_argument('--gap_name',type=str,default='data_all/xxd_good_and_m7_model_gap.csv',help='training data in csv format')
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
def good_ratio_plot(data, col_name,SP,monthdays,fig_save_path,m=2000):

    start_data = '2013-04-01 00:00:00'
    end_data = '2014-07-01 00:00:00'

    start_data = datetime.datetime.strptime(start_data, '%Y-%m-%d %H:%M:%S')
    end_data = datetime.datetime.strptime(end_data, '%Y-%m-%d %H:%M:%S')
    windows = (end_data - start_data).days / monthdays
    start='2013-04-01 00:00:00'
    end='2014-04-01 00:00:00'
    start = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    end = datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S')

    N_plot_list=[]
    ratio_plot_list=[]
    new_data = data[-data[col_name].isnull()]
    if len(new_data)<m:
        pass
    else:
        for i in range(windows + 1):
            data_notnull = new_data[new_data["issue_date"] >= str(start)]
            data_notnull = data_notnull[data_notnull["issue_date"] <= str(end)]
        #######################  数据排序  #######################
            index = np.argsort(data_notnull[col_name])
            sorted_col=list(pd.Series(list(data_notnull[col_name]))[index])#排序后的列值
            sorted_label=list(pd.Series(list(data_notnull['label_profit']))[index])#排序后的标签
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
            cur_cnt=len_list[0]
            for first_m in range(0,num_group):
                if(cur_cnt<m):
                    try:
                     cur_cnt=cur_cnt+len_list[first_m+1]
                    except:
                        continue
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
            N_plot_list.append(N)
            ratio_plot_list.append(ratio)
            start = start + datetime.timedelta(days=monthdays)
            end = end + datetime.timedelta(days=monthdays)
        #画出好人比例图
        for i in range(windows + 1):
            plt.plot(N_plot_list[i], ratio_plot_list[i],label=i)
        plt.legend()
        plt.savefig(str(fig_save_path)+"_"+SP+"_"+str(col_name) + '.png', dpi=180)
        plt.close()
        return max(ratio), min(ratio), max(ratio) - min(ratio)


#画离散图
def good_ratio_plot_discrete(new_data,colom,fig_save_path,monthdays,SP):
    """
        INPUT:
          dataframe: dataframe format. The first column must be label.
        OUTPUT:
          1)dictionary:
            key is feature name,
            value is a list which contains three parts: bin points(list), logodds(list), logodds for null(a number, if have null values)
          2)odds plot: if the x-axis has -1, that means bin for null values.
    """
    start_data = '2013-04-01 00:00:00'
    end_data = '2015-07-01 00:00:00'
    start_data = datetime.datetime.strptime(start_data, '%Y-%m-%d %H:%M:%S')
    end_data = datetime.datetime.strptime(end_data, '%Y-%m-%d %H:%M:%S')
    windows = (end_data - start_data).days / monthdays
    print windows
    start = '2013-04-01 00:00:00'
    end = '2014-04-01 00:00:00'
    start = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    end = datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S')


    fea_list = new_data.columns  # feature list, the first one is label
    output = {}
    N_plot_list = []
    ratio_plot_list = []

    new_data = new_data[-new_data[colom].isnull()]

    for i in range(windows + 1):
        data_notnull = new_data[new_data["issue_date"] >= str(start)]
        data_notnull = data_notnull[data_notnull["issue_date"] <= str(end)]


        col = list(data_notnull[colom])
        distinct_val = set(col)
        sorted_col = sorted(distinct_val)
        ratio = []
        for val in sorted_col:
            data_tmp = data_notnull[data_notnull[colom] == val]
            n_cur = len(data_tmp)
            n_good = len(data_tmp[data_tmp['label_profit'] == 1])
            ratio.append(n_good*1.0/n_cur)

        sorted_col_update=flatten([sorted_col[0],sorted_col[:]])
        sorted_col_update[0]=sorted_col_update[0]-0.5
        sorted_col_update[-1]=sorted_col_update[-1]+0.5

        for i in range(1, len(sorted_col_update) - 1):
            sorted_col_update[i] = (sorted_col_update[i] + sorted_col_update[i+1]) * 1.0 / 2

        output[colom] = [sorted_col_update, ratio]
        N_plot_list.append(sorted_col)
        ratio_plot_list.append(ratio)
        start = start + datetime.timedelta(days=monthdays)
        end = end + datetime.timedelta(days=monthdays)

        #############################################  logodds plot  #######################################################

    for i in range(windows + 1):
        plt.plot(N_plot_list[i], ratio_plot_list[i], label=i)

    plt.legend()
    plt.savefig(str(fig_save_path) + "_" + SP + "_" + str(colom) + '.png', dpi=180)
    plt.close()
    return max(ratio), min(ratio), max(ratio) - min(ratio)

def Plot_goodradio(new_data):


    Continuous = ["b.id_number","a.apply_id","b.transport_id","b.mortgagor_id","a.contract_no","mm","a.issue_date","a.m7_ratio","a.revenue","a.total_expense",
                    "a.profit","a.label","a.label_profit",0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,1,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                    ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
    monthdays = 180
    # gap_name=args.gap_name
    # gap = open(gap_name, 'wb+')
    # gap_output = csv.writer(gap, dialect='excel')
    # gap_output.writerow(["feature","max","min","gap"])

    fea_list = new_data.columns
    print fea_list

    for m in range(15, len(fea_list)-1):
        colom = fea_list[m]
        print colom
        print "it is Continuous or nor 0:Continuous 1:not",Continuous[m]

        fig_save_path = 'xxd_good_radio_windows_180/'
        # fig_save_path = str(fig_save_path) + str(colom) + '/'
        if not os.path.exists(fig_save_path):
            os.makedirs(fig_save_path)
            print "The path of saving figs has been created"
        if Continuous[m]==0:
            Max, Min, gap=good_ratio_plot(new_data,colom,"all",monthdays,fig_save_path)
        else:
            Max,Min,gap=good_ratio_plot_discrete(new_data,colom,fig_save_path,monthdays,"all")
        # gap_output.writerow([colom,Max,Min,gap])


if __name__ == '__main__':
    data_name = args.data_name
    data_name=load_data(data_name)
    Plot_goodradio(data_name)