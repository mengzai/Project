#-*- coding: UTF-8 -*-
import csv


import numpy as np



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


#转出:to_csv
def output_to_csv(names_value, output_file_name):
    with open(output_file_name, 'w') as file0:
        output = csv.writer(file0, dialect='excel')
        for item in names_value:
            output.writerow(item)

#转出:to_txt
def output_to_txt(names_value, output_file_name):
    with open(output_file_name, 'w') as output:
        for item in names_value:
            output.write('%s\t%s\n' % (item[0], item[1]))


#作用:按照根据某一列:第3列  降续排列
#input:[[1,2,4,5],[2,3,3,4]]
#output: 双层list
def result_sort(result,index=3,min=10):
    sort_list_index = []
    for line in result:
        sort_list_index.append(line[index])
    order = np.argsort(sort_list_index)
    order=order[::-1]
    final_sort_list = []
    for index in order:
        print float(result[index][2])
        if float(result[index][2])>=min:   ##选择某一列有限制
           final_sort_list.append(result[index])
    return final_sort_list

def main():
    #sort
    result=load_file('find_best_p.csv')
    sort_result=result_sort(result)
    # output_to_txt(sort_result, 'first_sort_train2.txt')
    output_to_csv(sort_result, 'find_sort_best_p.csv')

if __name__ == '__main__':
    main()
