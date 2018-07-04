# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import pickle
import math
import datetime
import numpy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_name',type=str,default='data/csxd_data/dh_csxd_train',help='training data in csv format')
parser.add_argument('--test_name',type=str,default='data/csxd_data/dh_csxd_test',help='training data in csv format')
parser.add_argument('--dh_all',type=str,default='data/all',help='training data in csv format')
parser.add_argument('--logistic_regression', type=str, default='logistic_regression',
                    help='training data in csv format')
# parser.add_argument('--term',type=str,default="version",help='training data in csv format')
args = parser.parse_args()

def calc_ods(bin_point_dict, dataframe):
    """
        This function calc odds given binpoints.
        INPUT:
          Dict: disctionary.
            key is feature name,
            value is a list which contains bin points
          dataframe: train data. The first column must be label.
          out_path : LOCATION to store the output
        OUTPUT:
          pickle(dataframe).The first column is label, followed by features(which in the same order of input)
        """
    import pandas as pd
    import numpy as np
    import math

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
        print col_name


    print output_dict



def find_spit():
   spit={
    'MAX_TOTAL_OVERDUE_DAYS':[0,0.1,34,50],
    'default_times':[0,0.1,1,20],
    'CREDIT_CARD_NUM':[0,0.1,1,3,6,13,40],
    'NOMAL_CARD_NUM':[0,0.1,1,2,5,6,10,40],
    'CLOSED_CARD_NUM':[0,0.1,2,3,4,10],
    'NON_ACTIVATE_CARD_NUM':[0,0.1,1,2,3,10],
    'CREDIT_CARD_MAX_NUM':[0,0.1,3,37,68,100],
    'LOAN_MAX_NUM':[1,1.2,4,15,32,72,200],
    'MAX_CREDIT_CARD_AGE':[0,0.1,1,34,79,200],
    'MAX_CREDIT_LINE':[0,0.1,1906,10000,16000,30000,50715,10000000],
    'MAX_LOAN_AGE':[1,1.1,15,39,97,200],
    'TOTAL_OVERDUE_NUM_L':[0,0.1,1,7,50],
    'QUERY_TIMES2':[0,0.1,7,50],
    'OVERDUE_CARD_NUM':[0,0.1,3,5,50],
    'MAX_OVERDUE_NUM_C':[0,0.1,3,11,50],
    'HOUSING_LOAN_NUM':[0,0.1,1,20],
    'contact_num':[4,7,50],
   }

if __name__ == '__main__':
    dh_all=args.dh_all
    col_name='CREDIT_CARD_MAX_NUM'
    data1=pd.read_csv(dh_all)
    data_notnull1=data1[-data1[col_name].isnull()]

    tmp1 = list(data_notnull1[col_name])
    data=pd.Series(sorted(tmp1))

    spit=[0,0.9]
    for  i in range(len(spit)):
        print data[int(len(tmp1)*spit[i]+1)]
    print data[[int(len(tmp1)*0+1),int(len(tmp1)*0.23),int(len(tmp1)*0.95),int(len(tmp1)*0.975),int(len(tmp1)*0.91),int(len(tmp1)-1)]]
