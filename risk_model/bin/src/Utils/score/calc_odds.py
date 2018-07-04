import pandas as pd
import numpy as np
import math

"""
This function calc odds given binpoints.
:dataframe: train data. The first column must be label.
:out_path : LOCATION to store the output
:return:
"""
def calc_ods(bin_point_dict, dataframe):

    fea_list = dataframe.columns  # feature list, the first one is label
    output_dict = {}
    for k in range(1, len(fea_list)):  # process one column each time
        col_name = fea_list[k]
        data = dataframe[[col_name, 'label']]
        data_notnull = data[-data[col_name].isnull()]
        index = np.argsort(list(data_notnull[col_name]))
        sorted_col = data_notnull.iloc[index, 0]
        sorted_col = list(sorted_col)
        label = data_notnull.iloc[index, 1]  # sort label in the same order as column value
        label = pd.Series(list(label))
        num_nonull = len(data_notnull)

        bin_point = bin_point_dict[col_name]
        pos = np.digitize(sorted_col, bin_point)
        bin=pd.Series(pos)
        bin[bin > len(bin_point) - 1] = len(bin_point) - 1  # if  point is larger than the last bin point, assign it to the last bin
        bin[bin < 1] = 1

        logodds=[]
        for i in range(1,len(bin_point)):
            cur_len=sum(bin==i)
            cur_good=sum(label[bin==i])

            if (cur_len==cur_good):  # if the bin only has good
                cur_odds = 9
            elif (cur_good == 0):  # if the bin only has bad
                cur_odds = -9
            else:
                odds_origin = math.log(cur_good * 1.0 / (cur_len - cur_good))
                cur_odds = min(max(-9, odds_origin), 9)
            logodds.append(cur_odds)

        output_dict[col_name]=[bin_point,logodds]

        if(len(data)-num_nonull>0):
            num_null = len(data) - num_nonull
            null_good = sum(data[data[col_name].isnull()]['label'])
            null_N_good = [num_null, null_good]
            if (null_N_good[0] == null_N_good[1]):  # if null only has good
                null_logodds = 9
            elif (null_N_good[1] == 0):  # if null only has bad
                null_logodds = -9
            else:
                null_logodds_origin = math.log(null_N_good[1] * 1.0 / (null_N_good[0] - null_N_good[1]))
                null_logodds = min(max(-9, null_logodds_origin), 9)
            output_dict[col_name].append(null_logodds)

    return output_dict

def main():
    data_sub = data[['lab1el', 'MAR_STATUS', 'JOB_POSITION']]
    dict_instruct = {'MAR_STATUS': [0, 1.5, 3.5, 4.5], 'JOB_POSITION': [0, 3.5, 4.5, 8.5, 12]}
    calc_ods(dict_instruct, data_sub)


if __name__ == '__main__':
    main()




