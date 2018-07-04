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


def trans_to_second(source_file, up_level):
    first_fea_matrix = np.mat(source_file)
    first_level = np.mat(up_level)
    names_value = []
    for axis_y_i in range(1, first_fea_matrix.shape[1]):
        for axis_y_j in range(axis_y_i + 1, first_fea_matrix.shape[1]):
            if(first_level[0,axis_y_i] != first_level[0,axis_y_j]):
                little_block_num = int(0)
                bad_num = int(0)
                for label in range(0, first_fea_matrix.shape[0]):
                    if (first_fea_matrix[label, axis_y_i] == '1' and first_fea_matrix[label, axis_y_j] == '1'):
                        little_block_num += 1
                        if (first_fea_matrix[label, 0] == '0'):
                            bad_num += 1
                if little_block_num == 0:
                    continue
                else:
                    little_block = 1
                    new_feature_name = '{0}.{2}_{1}.{3}'.format(
                        str(axis_y_i), str(axis_y_j), little_block, little_block)
                    ratio = bad_num / little_block_num
                    name_temp = [new_feature_name, ratio,bad_num]
                    names_value.append(name_temp)
            # print(axis_y_j)
        print(axis_y_i)
    return names_value


def main():
    data = load_file('test_multidim.txt')
    level = load_file('index.txt')
    print('done')
    result = trans_to_second(data,level)
    output_result(result, 'second_test.txt')


if __name__ == '__main__':
    main()
