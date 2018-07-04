#-*- coding: UTF-8 -*-
import csv
import numpy as np
#-*- coding: UTF-8 -*-

import argparse
from csv import reader

"""
load_data:
读取文件封装文件:
load_pd_data:读取:dataframe 文件
load_file:txt,csv

return [[],[],[]]  双层


output:

"""

#转出:to_csv
def output_data(names_value, output_file_name):
    if output_file_name[-3:] == "csv":
        with open(output_file_name, 'w') as file0:
            output = csv.writer(file0, dialect='excel')
            for item in names_value:
                output.writerow(item)

    elif output_file_name[-3:] == "txt":
        with open(output_file_name, 'w') as output:
            for item in names_value:
                for i in range(0,len(item)):
                    output.write('%s\t' % (item[i]))
                output.write('\n')
    else:
        print "此输出文件非txt 及csv,请以 txt or csv  结尾"


def load_file(filename):
    if filename[-3:]=="txt":
        with open(filename, 'r') as data_file:
            data = []
            for line in data_file.readlines():
                line = line.strip('\n').split('\t')
                data.append(line)
            return data
    if filename[-3:] == "csv":
        with open(filename, 'r') as data_file:
            data = []
            for line in data_file.readlines():
                line = line.strip('\r\n').split(',')
                data.append(line)
            return data
    else:
        print "此输入文件非txt 及csv,请以 txt or csv  结尾"





def find_best_p(source_file,pre_result,first_feature):

    first_fea_matrix = np.mat(source_file)

    names_value = []
    for axis_y_i in range(0, first_fea_matrix.shape[0]):
        sum_bin = 0.0
        sum_p = 0.0
        num=0

        for label in range(1, first_fea_matrix.shape[1]):
            axis_x_y=int(first_fea_matrix[axis_y_i,label])
            sum_bin +=axis_x_y
            if int(axis_x_y)==1:
                sum_p  +=float(pre_result[num][3])
            num += 1

        if sum_bin==0:
            sum_avg=first_feature[axis_y_i][3]
            print "err"

        else:
            sum_avg =sum_p*1.0/sum_bin
        name_temp = [axis_y_i,sum_p, sum_bin,sum_avg,first_fea_matrix[axis_y_i,0]]
        names_value.append(name_temp)
        print axis_y_i
    return names_value





