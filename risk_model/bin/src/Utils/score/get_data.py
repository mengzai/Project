# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import pickle
import math
import datetime
import numpy
import argparse
import csv
parser = argparse.ArgumentParser()
parser.add_argument('--data_name',type=str,default='data_ori/dh_csxd_train_new',help='training data in csv format')
parser.add_argument('--test_name',type=str,default='data_ori/dh_csxd_test_title',help='training data in csv format')
args = parser.parse_args()

def load_data(path):
    return pd.read_csv(path, error_bad_lines=False)

#转出:to_csv
def output_data(names_value, output_file_name):
    if output_file_name[-3:] == "csv":
        with open(output_file_name, 'w') as file0:
            output = csv.writer(file0, dialect='excel')
            for item in names_value:
                output.writerow(item)

    elif output_file_name[-3:] == "txt":
        with open(output_file_name, 'w') as output:
            for item in names_value:
                for i in range(0,len(item)):
                    output.write('%s\t' % (item[i]))
                output.write('\n')
    else:
        print "此输出文件非txt 及csv,请以 txt or csv  结尾"


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

def ues():
#     data_name=args.test_name
#     data=load_data(data_name)
#
#     clom=['label', 'APPLY_MAX_AMOUNT',  'JOB_POSITION', 'ACCEPT_MOTH_REPAY', 'SCORE',
#      'FEE_MONTHS','LONG_REPAYMENT_TERM','GENDER',
#      'apply_times', 'age', 'RECRUITMENT_DATE', 'IN_CITY_YEARS', 'pledge_num', 'n1', 'n5', 'end_balance', 'break_times',
#      'MAXOVER_DAYS',
#      'MAXOVER_AMOUNT', 'loan_purpose', 'mobile_type', 'ORG_TYPE', 'MAX_DIPLOMA', 'MARRIAGE', 'HAS_CAR',
#      'HOUSE_CONDITION','INDUSTRY1','risk_industry','IS_OVERDUE','MAX_TOTAL_OVERDUE_DAYS','default_times','CREDIT_CARD_NUM',
# 'NOMAL_CARD_NUM','CLOSED_CARD_NUM','NON_ACTIVATE_CARD_NUM','CREDIT_CARD_MAX_NUM','LOAN_MAX_NUM','MAX_CREDIT_CARD_AGE',
# 'MAX_CREDIT_LINE', 'MAX_LOAN_AGE','TOTAL_OVERDUE_NUM_L','QUERY_TIMES2','OVERDUE_CARD_NUM','MAX_OVERDUE_NUM_C','HOUSING_LOAN_NUM']
#
#     datanew=data[clom]
#     datanew.fillna(-999999, inplace = True)
    train=load_data('/Users/wanglili/Documents/workstation/working/fengxian_ML/GBM/test_GBM.csv')
    output_data(train,'/Users/wanglili/Documents/workstation/working/fengxian_ML/GBM/GBM_test.txt')
if __name__ == '__main__':
    ues()