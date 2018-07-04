from __future__ import division
from __future__ import print_function

import numpy as np


def load_file(filename):
    with open(filename, 'r') as data_file:
        data = []
        for line in data_file.readlines():
            line = line.strip('\n').split('\t')
            data.append(line)
        return data


def output_result(names_value, output_file_name):
    with open(output_file_name, 'w') as output:
        for item in names_value:
            output.write('%s\t%s\n' % (item[0], item[1]))


def trans_to_second(source_file):
    first_fea_matrix = np.mat(source_file)
    names_value = []
    for axis_y_i in range(1, first_fea_matrix.shape[1]):
        for axis_y_j in range(axis_y_i + 1, first_fea_matrix.shape[1]):
            little_block_num = {'00': 0, '01': 0, '10': 0, '11': 0}
            bad_num = {'00': 0, '01': 0, '10': 0, '11': 0}
            for label in range(0, first_fea_matrix.shape[0]):
                i_j_str = first_fea_matrix[label, axis_y_i] + first_fea_matrix[label, axis_y_j]
                little_block_num[i_j_str] += 1
                if first_fea_matrix[label, 0] == '0':
                    bad_num[i_j_str] += 1
            for little_block in bad_num.keys():
                if little_block_num[little_block] == 0:
                    continue
                else:
                    new_feature_name = '{0}.{2}_{1}.{3}'.format(
                        str(axis_y_i), str(axis_y_j), little_block[0], little_block[1])
                    ratio = bad_num[little_block]*1.0/ little_block_num[little_block]
                    name_temp = [new_feature_name, ratio]
                names_value.append(name_temp)
            # print(axis_y_j)
        print(axis_y_i)
    return names_value


def main():
    data = load_file('1.txt')
    print('done')
    result = trans_to_second(data)
    output_result(result, 'second_train.txt')

if __name__ == '__main__':
    main()
