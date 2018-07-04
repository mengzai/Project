#-*- coding: UTF-8 -*-
from  csv import reader
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from compiler.ast import flatten



import numpy as np


#读入
def load_file(filename):
    with open(filename, 'r') as data_file:
        data = []
        for line in data_file.readlines():
            line = line.strip('\r\n').split(',')
            data.append(line)
        return data

    # 读入
def load_csv(filename):
    data=reader(open(filename),'rb')
    data_rows=[d for d in data]
    return data_rows

#转出:to_csv
def output_to_csv(names_value, output_file_name):
    with open(output_file_name, 'w') as file0:
        output = csv.writer(file0, dialect='excel')
        for item in names_value:
            print item
            output.writerow(item)

#转出:to_txt
def output_to_txt(names_value, output_file_name):
    with open(output_file_name, 'w') as output:
        for item in names_value:
            output.write('%s\t%s\t%s\t%s\n' % (item[0], item[1],item[2], item[3]))


#作用:按照根据某一列:第3列  降续排列
#input:[[1,2,4,5],[2,3,3,4]]
#output: 双层list
def result_sort(result,index=3,min=100):
    sort_list_index = []
    for line in result:
        sort_list_index.append(line[index])
    order = np.argsort(sort_list_index)
    order=order[::-1]
    final_sort_list = []
    for index in order:
        if int(result[index][2])>=min:   ##选择某一列有限制
           final_sort_list.append(result[index])
    return final_sort_list



#input:result:[[],[]]每一行的具体值
#return
# index:为横坐标名称
#fianl_bad_radio,fianl_good_radio   坏人;好人累计百分比
def feature_ks(result):

    index=[]
    fianl_good_radio=[]
    fianl_bad_radio=[]
    all_man=0
    good_man=0
    bad_man=0

    for line in result:
        index.append(str(line[0])+'_'+str(line[1]))
        all_man += int(line[2])
        bad_man +=int(line[3])
        good_man += int(line[4])

        fianl_good_radio.append(float(good_man*1.0/all_man))
        fianl_bad_radio.append(float(bad_man * 1.0 / all_man))

    return index,fianl_bad_radio,fianl_good_radio


def main():
    #sort
    result=load_csv('data/AUC_result.csv')
    index, fianl_bad_radio, fianl_good_radio=feature_ks(result)
    # output_to_csv(sort_result, 'second_sort_train.csv')

if __name__ == '__main__':
    main()
