#-*- coding: UTF-8 -*-

import pandas as pd
import numpy
import matplotlib.pyplot as plt
import os


def load_data(table_name):
    data = pd.read_csv(table_name)
    return data

#table_name是数据集名称,col_name是要分析的列名,m是每组最小样本数(阈值),默认值为2000
def good_ratio_plot(data, col_name,SP,fig_save_path,m=2000):
    data_notnull = data[-data[col_name].isnull()]
    if len(data_notnull)<m:
        pass
    else:
        #######################  数据排序  #######################
        index = numpy.argsort(data_notnull[col_name])
        print index
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
        fig1.show()

        # # 画出密度图
        # fig2 = plt.figure(figsize=(10, 10))
        # plt.plot(N, density, 'r')
        # plt.xlabel(str(col_name))
        # plt.ylabel('density')
        # plt.xlim(0,1)
        # fig2.savefig(str(col_name)+ '_density.png', dpi=180)
# def denstiny(data, col_name,SP,fig_save_path):
#     data_notnull = data[-data[col_name].isnull()]
#     #######################  数据排序  #######################
#     # plt.hist(data_notnull[col_name], bins=50, color='steelblue', normed=True)
#     plt.plot(len(data_notnull[col_name]), data_notnull[col_name], 'r')
#     plt.show()


# # 用法:
import time
start = time.clock()
Dataset=load_data("data_all/xxd_good_and_m7_model")
good_ratio_plot(Dataset,'INDUSTRY1',"All","figure/")
end = time.clock()
print('Running time: %s Seconds' % (end - start))






