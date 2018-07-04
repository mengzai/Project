#-*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import numpy
import matplotlib.pyplot as plt



def load_data(filename):
    data = pd.read_csv(filename, error_bad_lines=False)
    return data
def split_bin_su(dataframe_my):
    import pandas as pd
    import math
    import numpy
    import pickle

    fea_list = dataframe_my.columns  # feature list, the first one is label
    output = {}  # output dictionary

    for k in range(1, len(fea_list)):  # process one column each time
        col_name = fea_list[k]

        data = dataframe_my[[col_name, 'label']]
        data_notnull = data[-data[col_name].isnull()]
        # sorted_col = sorted(data_notnull[col_name])#sort the column value asc
        index = numpy.argsort(data_notnull[col_name])
        sorted_col = data_notnull.iloc[index, 0]
        sorted_col = list(sorted_col)
        label = data_notnull.iloc[index, 1]  # sort label in the same order as column value
        label = list(label)

        ##########################################  bin_point  #####################################################
        # set the maximum number of bins
        num_bin = 10
        # minimum number of points in each bin
        min_num = int(len(data_notnull) * 1.0 / num_bin)
        # bin_point is bin points, the first bin point is the minimum point of the column value
        bin_point = [sorted_col[0]]
        # index1 is the location of bin point
        index1 = [0]
        i = 0

        while i < len(data_notnull):
            if (len(data_notnull) - i > min_num):
                i = i + min_num
                tmp = sorted_col[i]
                for j in range(i + 1, len(data_notnull)):
                    if (sorted_col[j] == tmp):
                        j = j + 1
                    else:
                        tmp = sorted_col[j - 1]
                        i = j - 1
                        index1.append(j - 1)
                        bin_point.append(tmp)
                        break
            else:
                break

        # if the last bin is too small, combine it with the previous one
        if (len(data_notnull) - 1 - index1[-1] < min_num and index1[-1] != len(data_notnull) - 1):
            bin_point.pop(-1)
            index1.pop(-1)
        # add the last point to the bin_point if the last point is not in the binpoint
        if (index1[-1] != len(data_notnull) - 1):
            index1.append(len(data_notnull) - 1)
            bin_point.append(sorted_col[-1])

        bin_point[0] = bin_point[0] - 0.5
        bin_point[-1] = bin_point[-1] + 0.5

        for i in range(1, len(bin_point) - 1):
            bin_point[i] = (sorted_col[index1[i]] + sorted_col[index1[i] + 1]) * 1.0 / 2

        ################################################## calc odds ##########################################################

        index1[0] = index1[0] - 1
        # group of value
        group = [sorted_col[index1[i] + 1: index1[i + 1] + 1] for i in range(0, len(index1) - 1)]
        # group of label
        group_label = [label[index1[i] + 1: index1[i + 1] + 1] for i in range(0, len(index1) - 1)]
        index1[0] = index1[0] + 1

        bin_N_good = [[len(group[i]), sum(group_label[i])] for i in range(0, len(index1) - 1)]
        logodds = []

        for i in range(0, len(index1) - 1):
            if (bin_N_good[i][1] == bin_N_good[i][0]):  # if the bin only has good
                cur_odds = 9
            elif (bin_N_good[i][1] == 0):  # if the bin only has bad
                cur_odds = -9
            else:
                odds_origin = math.log(bin_N_good[i][1] * 1.0 / (bin_N_good[i][0] - bin_N_good[i][1]))
                cur_odds = min(max(-9, odds_origin), 9)
            logodds.append(cur_odds)

        output[col_name] = [bin_point, logodds]

        if (sum(data[col_name].isnull()) > 0):  # if the column has null value
            null_N_good = [sum(data[col_name].isnull()),
                           sum(map(lambda x, y: x & y, data[col_name].isnull(), data['label'] == 1))]
            if (null_N_good[0] == null_N_good[1]):  # if null only has good
                null_logodds = 9
            elif (null_N_good[1] == 0):  # if null only has bad
                null_logodds = -9
            else:
                null_logodds_origin = math.log(null_N_good[1] * 1.0 / (null_N_good[0] - null_N_good[1]))
                null_logodds = min(max(-9, null_logodds_origin), 9)
            output[col_name].append(null_logodds)
    return output
def odds_transform_single(Dict, Dataframe_my,path):
    output_df = pd.DataFrame()

    output_df['label'] = Dataframe_my['label']

    # for k in range(1, len(fea_list)):
    for (key, val) in Dict.items():
        col_name = key
        print col_name
        bin_point = Dict[col_name][0]
        logodds = Dict[col_name][1]
        data = Dataframe_my[[col_name]]
        data_notnull = data[-data[col_name].isnull()]
        col = list(data_notnull[col_name])

        pos = np.digitize(col, bin_point)  # the bin number for each non-null point in the column
        bin = pd.Series([None] * len(data))
        bin[-data[col_name].isnull()] = pos
        bin[bin > len(bin_point) - 1] = len(
            bin_point) - 1  # if  point is larger than the last bin point, assign it to the last bin
        bin[bin < 1] = 1  # if  point is smaller than the first bin point, assign it to the first bin

        odds = pd.Series([None] * len(data))
        odds[-bin.isnull()] = [logodds[i - 1] for i in bin[-bin.isnull()]]  # assign logodds for each non-null point

        if (sum(data[col_name].isnull()) > 0):  # if has null value
            # print Dict[col_name]
            null_logodds = Dict[col_name][2]
            odds[data[col_name].isnull()] = null_logodds

        output_df[col_name] = odds

    # Path =  'data/odds/data_odds.pkl'
    # file = open(Path, 'wb')
    # pickle.dump(output_df, file)
    # file.close()

    # output csv format
    txtPath = path
    output_df.to_csv(txtPath, sep='\t', index=False, header=False)

    # profitpath=out_path +'/'+SP+ 'weight.txt'

    # new=dataframe[dataframe['label_profit']==0]['label_profit']
    # dataframe = pd.DataFrame(dataframe)

    # 将负值的特征
    # dataframe[dataframe['label_profit']==0] = 3
    # print dataframe['label_profit']
    # dataframe['label_profit'].to_csv(profitpath, sep='\t', index=False, header=False)
    #
    # 将收益作为权重
    # pd.DataFrame(map(abs, [(dataframe['profit']*0.0001)])).T.to_csv(profitpath, sep='\t', index=False, header=False)

    # 将负收益的2倍作为权重
    # print pd.DataFrame(np.array(dataframe[dataframe['profit']<0])*1.5)
    # dataframe[dataframe['profit']<0]=pd.DataFrame(np.array(dataframe[dataframe['profit']<0])*2)
    # pd.DataFrame(map(abs, [(dataframe['profit'] * 0.0001)])).T.to_csv(profitpath, sep='\t', index=False, header=False)
