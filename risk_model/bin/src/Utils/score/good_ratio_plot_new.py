#-*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import numpy
import matplotlib.pyplot as plt

#table_name是数据集名称,col_name是要分析的列名,m是每组最小样本数(阈值),默认值为2000
def good_ratio_plot(data, col_name, points=[], m=2000):

    data_notnull = data[-data[col_name].isnull()]
    # print len(data),len(data_notnull),len(data_notnull[data_notnull[col_name]>0.985])

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
    #density=[]#存储密度
    cur_cnt=len_list[0]
    for first_m in range(0,num_group-1):
        if(cur_cnt<m):
            cur_cnt=cur_cnt+len_list[first_m+1]
        else:
            break


    if(first_m%2==1):
        first_point=(first_m-1)/2
    if(first_m%2==0):
        first_point=first_m/2


    total_cnt1 = len_list[0:first_m+1]
    total_good1 = sum_list[0:first_m+1]
    ratio_1=sum(total_good1)*1.0/sum(total_cnt1)
    for i in range(0,first_point+1):
        ratio.append(ratio_1)
        #density.append(sum(total_cnt1) * 1.0 / (group_data[first_m][0] - group_data[0][0]))

    if(first_m%2==1 and first_m+1<=num_group-1):
        ratio.append((sum(total_good1)+sum_list[first_m+1])*1.0/(sum(total_cnt1)+len_list[first_m+1]))
        #density.append((sum(total_cnt1)+len_list[first_m+1]) * 1.0 / (group_data[first_m+1][0] - group_data[0][0]))
        first_point=first_point+1
        first_m=first_m+1



    forward_pointer = first_m
    backward_pointer = 0

    for i in range(first_point+1,num_group):
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
                    #density.append(sum(total_cnt_end) * 1.0 / (group_data[forward_pointer][0] - group_data[backward_pointer][0]))
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
                #density.append(sum(total_cnt) * 1.0 / (group_data[forward_pointer][0] - group_data[backward_pointer][0]))


    #画出好人比例图
    fig1 = plt.figure(figsize=(10, 10))
    plt.plot(N, ratio, 'b')
    plt.xlabel(str(col_name))
    plt.ylabel('good ratio')
    # plt.xlim(0.9,1)
    #y0 = sum(sorted_label)*1.0/len(sorted_label)
    #plt.axhline(y=y0, linewidth=0.3, color='k')

    for i in range(0,len(points)):
        plt.axvline(x=points[i], linewidth=0.3, color='r')

    fig1.savefig('fig/spilt/1'+str(col_name)+'_goodratio' + '.png', dpi=180)
    print col_name



#用法:

#all continuous feature
data=pd.read_csv('dh_csxd_train')
col=['APPLY_MAX_AMOUNT', 'LONG_REPAYMENT_TERM', 'JOB_POSITION', 'ACCEPT_MOTH_REPAY', 'SCORE',
     'FEE_MONTHS',
     'apply_times', 'age', 'RECRUITMENT_DATE', 'IN_CITY_YEARS', 'pledge_num', 'n1', 'n5', 'end_balance', 'break_times',
     'MAXOVER_DAYS',
     'MAXOVER_AMOUNT', 'GENDER', 'loan_purpose', 'mobile_type', 'ORG_TYPE', 'MAX_DIPLOMA', 'MARRIAGE', 'HAS_CAR',
     'HOUSE_CONDITION','INDUSTRY1','risk_industry','IS_OVERDUE','MAX_TOTAL_OVERDUE_DAYS','default_times','CREDIT_CARD_NUM',
'NOMAL_CARD_NUM','CLOSED_CARD_NUM','NON_ACTIVATE_CARD_NUM','CREDIT_CARD_MAX_NUM','LOAN_MAX_NUM','MAX_CREDIT_CARD_AGE',
'MAX_CREDIT_LINE', 'MAX_LOAN_AGE','TOTAL_OVERDUE_NUM_L','QUERY_TIMES2','OVERDUE_CARD_NUM','MAX_OVERDUE_NUM_C','HOUSING_LOAN_NUM',
     'contact_num']

mydict_first={
    'MAXOVER_DAYS':[0,0.952,0.961,0.979,0.991],
    'MAXOVER_AMOUNT':[0,0,0.794,0.838,0.875,0.923,0.940,1],
    'MAX_TOTAL_OVERDUE_DAYS':[0,0.925,0.955,0.972],
    'default_times':[0,0.825,0.987],
    'CREDIT_CARD_NUM':[0,0.3,0.449,0.635,0.783,0.822,0.927],
    'NOMAL_CARD_NUM':[0,0.37,0.516,0.68,0.984],
    'CLOSED_CARD_NUM':[0,0.827,0.909,0.938,0.987],
    'NON_ACTIVATE_CARD_NUM':[0,0.906,0.962,0.984],
    'CREDIT_CARD_MAX_NUM':[0,0.232,0.295,0.5,0.778,0.9],
    'LOAN_MAX_NUM':[0,0.019,0.086,0.2665,0.6187,0.7868,0.902],
    'MAX_CREDIT_CARD_AGE':[0,0.236,0.3,0.507,0.719,0.9],
    'MAX_CREDIT_LINE':[0,0.262,0.3,0.50,0.59,0.689,0.794,0.903]}


mydict_second={
    'MAXOVER_DAYS':[0.0,0.1, 40.0, 68.0, 158.0, 319.0,500],
    'MAX_TOTAL_OVERDUE_DAYS':[0.0,0.1, 29.0, 93.0, 213.0,100000],
    'default_times':[0.0,0.1, 1.2, 2.2,100],
    'CREDIT_CARD_NUM':[0.0,0.1, 1.2, 3.2, 5.2, 8.2, 9.2, 14.2,50],
    'NOMAL_CARD_NUM':[0.0,0.1, 1.2, 3.2, 5.2, 18.2,50],
    'CLOSED_CARD_NUM':[[0.0,0.1, 1.2, 3.2, 4.2, 8.2,50]],
    'NON_ACTIVATE_CARD_NUM':[0.0,0.1, 1.2, 3.2, 4.2,50],
    'CREDIT_CARD_MAX_NUM':[0.0,0.1, 1.2, 8.2, 34.2, 78.2, 96.2,200],
    'LOAN_MAX_NUM':[1.0,1.1, 2.2, 7.2, 22.2, 68.2, 98.2, 126.2,200],
    'MAX_CREDIT_CARD_AGE':[0.0,0.1, 2.0, 9.0, 35.0, 68.0, 96.0,200],
    'MAX_CREDIT_LINE':[0.0,0.1, 500.0, 3000.0, 10000.0, 15000.0, 20000.0, 31000.0, 52000.0,1000000000]}



default_times_1 165

CREDIT_CARD_NUM_1 170

NOMAL_CARD_NUM_1 179

CLOSED_CARD_NUM_1 186

NON_ACTIVATE_CARD_NUM_1 193

CREDIT_CARD_MAX_NUM_1 199

LOAN_MAX_NUM_1 207


TOTAL_OVERDUE_NUM_L_1 241
# col=['APPLY_MAX_AMOUNT', 'LONG_REPAYMENT_TERM']
# points=[[0.5,0.6],[0.5,0.6]]
# for key in mydict.keys():
#     good_ratio_plot(data,key,0)

# good_ratio_plot(data,'MAX_CREDIT_LINE',[0.262,0.3,0.50,0.59,0.689,0.794,0.903])


for key in mydict_first.keys():
       print key
       data_notnull = data[-data[key].isnull()]
       tmp1 = list(data_notnull[key])
       length=len(tmp1)
       print length
       list1=list(data_notnull)

       sort_data=pd.Series(sorted(tmp1))
       per_list = list(np.floor(np.array(mydict_first[key]) * length))
       print mydict_first[key]
       print sort_data[per_list]
       print list(sort_data[per_list])




# col_name='TOTAL_OVERDUE_NUM_L'
# data1=pd.read_csv('dh_csxd_train')
# data_notnull1=data1[-data1[col_name].isnull()]
# tmp1 = list(data_notnull1[col_name])
# length=len(tmp1)
# input=[0,0.86,0.895,0.942,0.963,0.984,1]
# data=pd.Series(sorted(tmp1))
# per_list=list(np.floor(np.array(input)*length))
# print data[per_list]

