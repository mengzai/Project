#-*- coding: UTF-8 -*-

import argparse
import math
import matplotlib.pyplot as plt
from compiler.ast import flatten
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--data_name',type=str,default='xxd_good_and_m7_model',help='training data in csv format')
args = parser.parse_args()
def good_ratio_plot_discrete(dataframe,fig_save_path,SP):
    """
        This function split bins for discrete feature.
        Each value is a bin.
        It can process one column or more than one column.
        INPUT:
          dataframe: dataframe format. The first column must be label.
        OUTPUT:
          1)dictionary:
            key is feature name,
            value is a list which contains three parts: bin points(list), logodds(list), logodds for null(a number, if have null values)
          2)odds plot: if the x-axis has -1, that means bin for null values.
    """

    fea_list = dataframe.columns  # feature list, the first one is label
    output = {}


    for k in range(1, len(fea_list)):
        col_name = fea_list[k]
        data = dataframe[[col_name, 'label_profit']]
        data_notnull = data[-data[col_name].isnull()]
        col = list(data_notnull[col_name])
        distinct_val = set(col)
        sorted_col = sorted(distinct_val)
        ratio = []
        for val in sorted_col:
            data_tmp = data[data[col_name] == val]
            n_cur = len(data_tmp)
            n_good = len(data_tmp[data_tmp['label_profit'] == 1])
            ratio.append(n_good*1.0/n_cur)

        sorted_col_update=flatten([sorted_col[0],sorted_col[:]])
        sorted_col_update[0]=sorted_col_update[0]-0.5
        sorted_col_update[-1]=sorted_col_update[-1]+0.5
        for i in range(1, len(sorted_col_update) - 1):
            sorted_col_update[i] = (sorted_col_update[i] + sorted_col_update[i+1]) * 1.0 / 2
        output[col_name] = [sorted_col_update, ratio]


        #############################################  logodds plot  #######################################################

        fig = plt.figure(figsize=(10, 10))
        plt.plot(sorted_col, ratio, 'ro')
        plt.plot(sorted_col, ratio, 'b')
        plt.xlabel('values')
        plt.ylabel('good ratio')
        plt.xticks(sorted_col)
        # average level
        #y0 = math.log(sum(data['label'] == 1) * 1.0 / (len(data) - sum(data['label'] == 1)))
        y0=sum(data_notnull['label_profit'] == 1) * 1.0 / len(data_notnull)
        #y0=0.5
        plt.axhline(y=y0, linewidth=0.3, color='k')
        fig.savefig(str(fig_save_path)+"_"+SP+"_"+str(col_name) + '.png', dpi=180)

    print  output


def load_data(path):
    return pd.read_csv(path, error_bad_lines=False)

if __name__ == '__main__':
    data_name = args.data_name
    new_data = load_data(data_name)
    dataFrame = new_data[['label_profit', "INDUSTRY1"]]
    good_ratio_plot_discrete(dataFrame,"figure","all")