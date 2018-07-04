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
            output.write('%s\n' % '\t'.join(map(lambda x:str(x),item)))

def output_result2(names_value, output_file_name):
    with open(output_file_name, 'w') as output:
        for item in names_value:
            output.write('%s\n' % '\t'.join(item))


def add_feature(source_file, add_feature):
    fea_matrix = np.mat(source_file)
    add_fea_index = np.mat(add_feature)
    final_matrix = np.zeros((fea_matrix.shape[0], 1), dtype=np.int)
    for add_index in range(0, add_fea_index.shape[0]):
        temp_fea_matrix = np.zeros((fea_matrix.shape[0],1),dtype=np.int)
        for line in range(0,fea_matrix.shape[0]):
            if (fea_matrix[line,int(add_fea_index[add_index,0])+1] == '1' and fea_matrix[line,int(add_fea_index[add_index,1])+1] == '1'):
                temp_fea_matrix[line,0] = int(1)
        final_matrix = np.c_[final_matrix,temp_fea_matrix]
    final_matrix = np.delete(final_matrix, 0, 1)
    # feature = final_matrix.tolist()
    return final_matrix

def delete_fir_fea(source_file, save_feature):
    fea_matrix = np.mat(source_file)
    save_feature = np.mat(save_feature)
    final_fea_matrix = fea_matrix[:,0]
    for add_index in range(0, len(save_feature)):
        temp_fea_matrix = fea_matrix[:, int(save_feature[add_index, 0])]
        final_fea_matrix = np.c_[final_fea_matrix,temp_fea_matrix]
    # np.delete(final_fea_matrix, 0, 1)
    # feature = final_fea_matrix.tolist()
    return final_fea_matrix

def compare(index,compare):
    result = []
    for item in index:
        for item2 in compare:
            if item[0] == item2[0]:
                result.append(item2)
    return result

def main():
    # data = load_file('train_multidim2.txt')
    # feature = load_file('add_second.txt')
    # save_feature = load_file('save_first.txt')
    # print('done')
    # result_part2 = add_feature(data,feature)
    # result_part1 = delete_fir_fea(data,save_feature)
    # result = np.c_[result_part1,result_part2].tolist()
    # output_result(result, 'train_after.txt')

    index = load_file('test_after.txt')
    print (len(index[0]))
    # compare2 = load_file('compare.txt')
    # result = compare(index,compare2)
    # output_result2(result,'first_value')

if __name__ == '__main__':
    main()
