#-*- coding: UTF-8 -*-
import csv


import numpy as np


#读入
def load_file(filename):
    with open(filename, 'r') as data_file:
        data = []
        for line in data_file.readlines():
            line = line.strip('\n').split(',')
            data.append(line)
        return data

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


def main():
    #sort
    result=load_file('result-1.csv')
    sort_result=result_sort(result)
    output_to_txt(sort_result, 'second_sort_train.txt')
    output_to_csv(sort_result, 'second_sort_train.csv')

if __name__ == '__main__':
    main()
