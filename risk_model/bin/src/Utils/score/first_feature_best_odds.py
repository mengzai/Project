#-*- coding: UTF-8 -*-
import csv


import numpy as np


def load_file(filename):
    with open(filename, 'r') as data_file:
        data = []
        bad = 0
        num=0

        for line in data_file.readlines():
            line = line.strip('\n').split('\t')
            num+=1
            if line[0]=='0':
                bad+=1
            data.append(line)


        print bad,num
        return data

def output_result(names_value, output_file_name):
    with open(output_file_name, 'w') as output:
        for item in names_value:
            output.write('%s\t%s\t%s\t%s\n' % (item[0], item[1],item[2],item[3]))

def trans_to_second(source_file):
    first_fea_matrix = np.mat(source_file)
    names_value = []
    for axis_y_i in range(1, first_fea_matrix.shape[1]):  #列
        little_block_num = { '1': 0}
        bad_num = {'1':0}
        for label in range(0, first_fea_matrix.shape[0]): #行
            i_str = first_fea_matrix[label, axis_y_i]
            if bad_num.has_key(i_str):
                little_block_num[i_str] += 1
                if first_fea_matrix[label, 0] == '0':
                    bad_num[i_str] += 1
        little_block='1'
        new_feature_name = str(axis_y_i)
        ratio = -1
        if little_block_num[little_block] == 0:
            print  axis_y_i
            pass
        else:
            ratio = bad_num[little_block] *1.0/ little_block_num[little_block]
            name_temp = [new_feature_name, ratio,bad_num[little_block],little_block_num[little_block]]
            names_value.append(name_temp)
            # print axis_y_i
    return names_value

def result_sort():
    sort_list_index = []
    for line in list:
        sort_list_index.append(line[1])

    order = np.argsort(sort_list_index)
    final_sort_list = []
    for index in order:
        final_sort_list.append(sort_list_index[index])
    return final_sort_list

def main():
    data = load_file('train_multidim2.txt')
    print('done')
    result = trans_to_second(data)
    output_result(result, 'first_train2.txt')

if __name__ == '__main__':
    main()
