#-*- coding: UTF-8 -*-
import csv
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--data_name',type=str,default='csxd_data/clic_csxd_train',help='training data in csv format')
parser.add_argument('--test_name',type=str,default='csxd_data/clic_csxd_test',help='training data in csv format')
parser.add_argument('--gap_name',type=str,default='csxd_data/clic_csxd_train_gap.csv',help='training data in csv format')
args = parser.parse_args()

#针对连续型变量
#table_name是数据集名称,col_name是要分析的列名,m是每组最小样本数(阈值),默认值为2000
def good_ratio_plot(data, col_name, m=2000):
    data_notnull = data[-data[col_name].isnull()]

    #######################  数据排序  #######################
    index = numpy.argsort(data_notnull[col_name])
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
    cur_cnt=len_list[0]
    for first_m in range(0,num_group-1):
        if(cur_cnt<m):
            cur_cnt=cur_cnt+len_list[first_m+1]
        else:
            break

    #print 'first_m:',first_m

    if(first_m%2==1):
        first_point=(first_m-1)/2
    if(first_m%2==0):
        first_point=first_m/2
    #print 'first_point:',first_point


    total_cnt1 = len_list[0:first_m+1]
    total_good1 = sum_list[0:first_m+1]
    ratio_1=sum(total_good1)*1.0/sum(total_cnt1)
    for i in range(0,first_point+1):
        ratio.append(ratio_1)

    if(first_m%2==1 and first_m+1<=num_group-1):
        ratio.append((sum(total_good1)+sum_list[first_m+1])*1.0/(sum(total_cnt1)+len_list[first_m+1]))
        first_point=first_point+1
        first_m=first_m+1


    #print 'first ratio:', ratio

    forward_pointer = first_m
    backward_pointer = 0

    for i in range(first_point+1,num_group):
        #print 'i:',i
        if(len_list[i]>=m):
            ratio.append(sum_list[i]*1.0/len_list[i])
            #density.append(len_list[i] * 1.0 / 0.1)
            forward_pointer=i
            backward_pointer=i
        else:
            if(forward_pointer==num_group -1):
                total_cnt_end = len_list[backward_pointer: forward_pointer + 1]  # 把延伸后的数据合成一个组
                total_good_end = sum_list[backward_pointer: forward_pointer + 1]
                ratio_end = sum(total_good_end) * 1.0 / sum(total_cnt_end)
                for j in range(i, forward_pointer + 1):
                    ratio.append(ratio_end)
                #print backward_pointer,forward_pointer
                break
            else:
                if(backward_pointer!=forward_pointer):
                    if (len_list[forward_pointer + 1] > len_list[backward_pointer] or sum(len_list[backward_pointer + 1: forward_pointer + 2]) >= m):
                        forward_pointer = forward_pointer + 1
                        backward_pointer = backward_pointer+1
                    else:
                        forward_pointer = forward_pointer + 2 if forward_pointer + 2 <= num_group - 1 else num_group - 1
                else:
                    forward_pointer = i + 1 if i + 1 <= num_group - 1 else num_group - 1

                total_cnt = len_list[backward_pointer: forward_pointer + 1]  # 把延伸后的数据合成一个组
                total_good = sum_list[backward_pointer: forward_pointer + 1]
                ratio.append(sum(total_good) * 1.0 / sum(total_cnt))  #计算好人比例

    return [sorted(set(sorted_col)), ratio]


#针对离散型变量
#输入是dataframe, 第一列为label,后面为feature
def good_ratio_plot_discrete(dataframe,col_name):
    """
        This function split bins for discrete feature.
        Each value is a bin.
        It can process one column or more than one column.
        INPUT:
          dataframe: dataframe format. The first column must be label.
        OUTPUT:
          1)dictionary:
            key is feature name,
            value is a list which contains three parts: bin points(list), logodds(list), logodds for null(a number, if have null values)
          2)odds plot: if the x-axis has -1, that means bin for null values.
    """

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

    return [sorted_col, ratio]



def good_ratio_plot_2line(train_name,test_name,col_name,line_gap_list):
    data1=pd.read_csv(train_name)
    data2=pd.read_csv(test_name)
    result1=good_ratio_plot(data1,col_name)
    result2 = good_ratio_plot(data2, col_name)
    fig1 = plt.figure(figsize=(10, 10))
    plt.plot(result1[0], result1[1], 'b',label='train')
    plt.plot(result2[0], result2[1], 'r',label='test')
    #############################改这里!!!!!!!#############################
    plt.xlim(line_gap_list[0], line_gap_list[1])                ################
    plt.xlabel(str(col_name))
    plt.ylabel('good ratio')
    plt.legend()
    fig1.savefig(str(col_name) + '_goodratio' + '.png', dpi=180)
    print col_name


def good_ratio_plot_2line_discrete(train_name,test_name,col_name):
    data1 = pd.read_csv(train_name)
    data2 = pd.read_csv(test_name)
    result1 = good_ratio_plot_discrete(data1, col_name)
    result2 = good_ratio_plot_discrete(data2, col_name)
    fig1 = plt.figure(figsize=(10, 10))
    plt.plot(result1[0], result1[1], 'b',label='train')
    plt.plot(result2[0], result2[1], 'r',label='test')
    plt.xlabel(str(col_name))
    plt.ylabel('good ratio')
    plt.legend()
    fig1.savefig(str(col_name) + '_goodratio' + '.png', dpi=180)
    print col_name

def load_data(path):
    return pd.read_csv(path, error_bad_lines=False)

# def Plot_goodradio():
#
#     gap_name=args.gap_name
#     gap = open(gap_name, 'wb+')
#     gap_output = csv.writer(gap, dialect='excel')
#     gap_output.writerow(["feature","max","min","gap"])
#     data_name = args.data_name
#     new_data = load_data(data_name)
#     # fea_list = new_data.columns
#     # print fea_list
#     for m in range(12, len(Continuous)):
#         colom = Continuous[m]
#         print colom
#
#
#         fig_save_path='fig_new_ratio/'
#         # fig_save_path = str(fig_save_path) + str(colom) + '/'
#         if not os.path.exists(fig_save_path):
#             os.makedirs(fig_save_path)
#             print "The path of saving figs has been created"
#
#         try:
#             if Continuous[m]==0:
#                 print "it is Continuous"
#                 Max, Min, gap =good_ratio_plot(new_data,colom,"all",fig_save_path)
#             else:
#                 print "it is not Continuous"
#                 dataFrame = new_data[['label_profit', colom]]
#                 Max,Min,gap=good_ratio_plot_discrete(dataFrame,fig_save_path,"all")
#                 print Max,Min,gap
#             gap_output.writerow([colom,Max,Min,gap])
#         except:
#             print "err"


##########################################################   用法   ##########################################################
def main():
    Continuous = ['MONTH_PAYMENT', 'RECRUITMENT_DATE', 'PROVIDE_FOR_COUNT', 'CREDIT_CARD_NUM', 'NOMAL_CARD_NUM',
                  'CLOSED_CARD_NUM',
                  'NON_ACTIVATE_CARD_NUM', 'CREDIT_CARD_MAX_NUM', 'LOAN_MAX_NUM', 'MAX_CREDIT_CARD_AGE',
                  'TOTAL_OVERDUE_NUM_C', 'MAX_LOAN_AGE'
                                         'TOTAL_OVERDUE_NUM_L', 'MAX_OVERDUE_NUM_C', 'MAX_OVERDUE_NUM_L',
                  'OVERDUE_CARD_NUM90_DAYS', 'OVERDUE_LOAN_NUM90_DAYS']

    Continuous1 = {'MONTH_PAYMENT':[0,100000000],
                  'RECRUITMENT_DATE':[0,45],
                  'PROVIDE_FOR_COUNT':[1,10],
                  'CREDIT_CARD_NUM':[0 ,50],
                  'NOMAL_CARD_NUM':[0 ,50],
                  'CLOSED_CARD_NUM':[0 ,50],
                  'NON_ACTIVATE_CARD_NUM':[0 ,50],
                  'CREDIT_CARD_MAX_NUM' :[0,450],
                  'LOAN_MAX_NUM':[0,450],
                  'MAX_CREDIT_CARD_AGE':[0,450],
                  'TOTAL_OVERDUE_NUM_C':[0,450],
                   'MAX_LOAN_AGE':[0,450],
                   'TOTAL_OVERDUE_NUM_L':[0,450],
                   'MAX_OVERDUE_NUM_C':[0,450],
                   'MAX_OVERDUE_NUM_L':[0,450],
                   'OVERDUE_CARD_NUM90_DAYS':[0,450],
                   'OVERDUE_LOAN_NUM90_DAYS':[0,450]
                  }

    data_name = args.data_name
    test_name = args.test_name
    print data_name
    # good_ratio_plot_2line('good_1.csv','test_24.csv','APPLY_MAX_AMOUNT')
    # good_ratio_plot_2line('good_1.csv', 'test_24.csv', 'GENDER')


    #连续型变量

    for i in range(0,len(Continuous)):
        print Continuous[i]
        # import time
        # start = time.clock()
        good_ratio_plot_2line(data_name, test_name, Continuous[i],Continuous1[Continuous[i]])
        # end = time.clock()
        # print('Running time: %s Seconds' % (end - start))



        # 离散型变量
    # col = ['LOAN_TYPE', 'loan_purpose', 'GENDER']
    # for i in range(0, len(col)):
    #     good_ratio_plot_2line_discrete('good_1.csv', 'test_24.csv', col[i])


if __name__=="__main__":
    main()







