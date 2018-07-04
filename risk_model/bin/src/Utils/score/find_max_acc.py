#-*- coding: UTF-8 -*-
import csv
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import argparse
import numpy as np
import math
import os
parser = argparse.ArgumentParser()
parser.add_argument('--predict_name',type=str,default='result_1.txt',help='training data in csv format')
args = parser.parse_args()

def load_data(path):
    return pd.read_csv(path, error_bad_lines=False)


'''
作用: 读取原文件数据,并且读取总人数,总好人数,总坏人数,将预测值排序order
:return:number 数据总数,
        good_man_all:总的好人数,
        bad_man_all,总坏人数
        order:排序之后的数据
        pre_dict:根据index找到:   label,pro
        pro_list:预测值的list
'''
def read_original_file():
    predict_name = args.predict_name

    good_man_all = 0  # 总的好人
    bad_man_all = 0  # 总的坏人
    number = 0  # 总人数
    pre_dict = {}  # 索引作为key,将label作为字典值,
    pro_list = []  # 预测值放入list中,排序之后并且找到索引

    file = open(predict_name, 'rb')
    for line in file.readlines():
        line = line.split("\t")

        pro_list.append(float(line[2]))
        pre_dict[number] = [int(line[0]), float(line[2])]  #label,pro

        number += 1  # 总数据数目
        if int(line[0]) == 1:
            good_man_all += 1  # 总好人
        if int(line[0]) == 0:
            bad_man_all += 1  # 总坏人

    order = np.argsort(pro_list)
    print "总好人数:",good_man_all
    print "总坏人数:",bad_man_all
    print "总人数:",number
    return number,good_man_all,bad_man_all,order,pre_dict,pro_list

'''
作用: 找到准确率最高时的分割点是什么
:读入:number 数据总数,
        good_man_all:总的好人数,
        order:排序之后的数据
        pre_dict:根据index找到:   label,pro
        pro_list:预测值的list
return: "准确率最高为:", max_fianl_pre
        "准确率最高时分割点", find_best_pre[max_fianl_pre]


'''
def find_the_highest_accuracy(number, good_man_all,bad_man_all, order,pre_dict,pro_list):
    pre_bad_man = 0  # 预测为坏人的数目
    true_bad_man = 0  # 真实为坏人的数目
    true_good_man = 0  # 真实为好人的数目
    max_fianl_pre = 0.0  # 最佳分割点,使得预测准确率最大
    find_best_pre = {}  # 通过字典索引值找到分割点

    for pre_val in order:
        dict_val = pre_dict[pre_val]
        if int(dict_val[0]) == 0:
            true_bad_man += 1
        else:
            true_good_man += 1

        pre_bad_man += 1

        badpre = true_bad_man * 1.0  # 真实的坏人
        if number - pre_bad_man == 0:
            goodpre = 0
        else:
            goodpre = (good_man_all - true_good_man) * 1.0  # (总的好人-bin之前的好人)剩余的好人
        fianl_pre = float((badpre + goodpre) * 1.0 / number)

        find_best_pre[fianl_pre] = pro_list[pre_val]

        if fianl_pre > max_fianl_pre:
            max_fianl_pre = fianl_pre

    print "准确率最高为:", max_fianl_pre
    print "准确率最高时分割点", find_best_pre[max_fianl_pre]
    return max_fianl_pre,find_best_pre

"""
坏人准确率最高
"""
def Find_the_maximum_recall(order,pro_list,bad_man_all,pre_dict,acc=0.00003):
    pre_bad_man = 0  # 预测为坏人的数目
    true_bad_man = 0  # 真实为坏人的数目

    spit_num=0.0
    for pre_val in order:
        dict_val = pre_dict[pre_val]
        if int(dict_val[0]) == 0:
            true_bad_man += 1
        pre_bad_man += 1
        badpre = true_bad_man * 1.0/pre_bad_man  #坏人的准确率
        if badpre>=acc:
            spit_num=pre_val
            break
        if spit_num==0.0:
            true_bad_man=0

    print "准确率大于:",acc,"时的分割点是",spit_num
    print "此分割点时找到的坏人是",true_bad_man ,"总坏人是",bad_man_all,"召回率",true_bad_man*1.0/bad_man_all

def find_max_cc():
    # number 数据总数, good_man_all:总的好人数, bad_man_all:总的坏人数,order:排序之后的数据 ,pre_dict:根据index找到:   label, pro;pro_list:预测值的list
    number, good_man_all,bad_man_all, order,pre_dict,pro_list=read_original_file()

    max_fianl_pre, find_best_pre=find_the_highest_accuracy(number, good_man_all,bad_man_all, order,pre_dict,pro_list)

    Find_the_maximum_recall(order,pro_list,bad_man_all,pre_dict)




if __name__=="__main__":
    find_max_cc()

