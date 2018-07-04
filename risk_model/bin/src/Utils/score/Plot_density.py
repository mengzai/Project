#-*- coding: UTF-8 -*-
import csv
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
parser = argparse.ArgumentParser()
parser.add_argument('--data_name',type=str,default='data_all/xxd_good_and_m7_model',help='training data in csv format')
parser.add_argument('--data_nameA',type=str,default='data_all/2013.csv',help='training data in csv format')
parser.add_argument('--data_nameB',type=str,default='data_all/2014.csv',help='training data in csv format')
parser.add_argument('--data_nameC',type=str,default='data_all/2015.csv',help='training data in csv format')
parser.add_argument('--data_nameD',type=str,default='data_all/2016.csv',help='training data in csv format')
args = parser.parse_args()


"""
ABCD 划分方式
A:true ok   predict ok
B:true ok   predict M7
C:true M7   predict ok
DB:true M7   predict M7
"""

def outliers_detection(data, times = 7, quantile = 0.95):
    data=data[-data.isnull()]
    data = np.array(sorted(data))
    #std-outlier
    outlier1 = np.mean(data) + 1*np.std(data)

    # mad-outlier
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    outlier2 = med + times * mad

    # quantile-outlier
    outlier3 = data[int(np.floor(quantile * len(data)) - 1)]
    return outlier1, outlier2, outlier3

def density(data1,data2,data3,data4,col_name,SP,outliers_max,fig_save_path):
    data_notnull1 = data1[-data1[col_name].isnull()][col_name]
    data_notnull2 = data2[-data2[col_name].isnull()][col_name]
    data_notnull3 = data3[-data3[col_name].isnull()][col_name]
    data_notnull4 = data4[-data4[col_name].isnull()][col_name]


    data_not_outliers1=data_notnull1[data_notnull1<=outliers_max]
    data_not_outliers2 = data_notnull2[data_notnull2<=outliers_max]
    data_not_outliers3 = data_notnull3[data_notnull3<=outliers_max]
    data_not_outliers4 = data_notnull4[data_notnull4<=outliers_max]

    # listab = [data_not_outliers1, data_not_outliers2]
    # listcd = [data_not_outliers3, data_not_outliers4]
    # listabcd=[data_not_outliers1,data_not_outliers2,data_not_outliers3, data_not_outliers4]
    # data_not_outliersab=pd.concat(listab)
    # data_not_outlierscd=pd.concat(listcd)
    # data_not_outliersabcd=pd.concat(listabcd)


    sns.distplot(data_not_outliers1, rug=True, hist=False, label='2013',color = "b")
    sns.distplot(data_not_outliers2, rug=True, hist=False, label='2014',color = "r")
    sns.distplot(data_not_outliers3, rug=True, hist=False, label='2015',color = "y")
    sns.distplot(data_not_outliers4, rug=True, hist=False, label='2016',color = "g")
    plt.savefig(str(fig_save_path) + "_" + SP + "_" + str(col_name) +'_single_'+ '.png', dpi=180)
    # plt.show()
    plt.close()


    # sns.distplot(data_not_outliersab, rug=True, hist=False, label='AB')
    # sns.distplot(data_not_outlierscd, rug=True, hist=False, label='CD')
    # sns.distplot(data_not_outliersabcd, rug=True, hist=False, label='ABCD')
    # plt.savefig(str(fig_save_path) + "_" + SP + "_" + str(col_name) +'_all'+ '.png', dpi=180)
    # # plt.show()
    # plt.close()

def load_data(path):
    return pd.read_csv(path, error_bad_lines=False)


def plot_density():

    #连续与否:  0:连续   1:不连续
    Continuous = ["b.id_number", "a.apply_id", "b.transport_id", "b.mortgagor_id", "a.contract_no", "a.issue_date",
                  "a.m7_ratio", "a.revenue", "a.total_expense",
                  "a.profit", "a.label", "a.label_profit", 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0,
                  1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0]

    data_name = args.data_name
    new_data = load_data(data_name)
    fea_list = new_data.columns

    data_nameA=args.data_nameA
    data_nameB = args.data_nameB
    data_nameC = args.data_nameC
    data_nameD = args.data_nameD

    Dataset1 = load_data(data_nameA)
    Dataset2 = load_data(data_nameB)
    Dataset3 = load_data(data_nameC)
    Dataset4 = load_data(data_nameD)

    test_feature=['GENDER', 'grade_version','HIGHEST_DIPLOMA','no_interrupted_card_num','card_interrupt', 'loan_purpose', 'n3',
                  'LONG_REPAYMENT_TERM', 'age', 'APPLY_MAX_AMOUNT','ACCEPT_MOTH_REPAY', 'credit_grade',  'in_city_years',
                   'MAX_CREDIT_CARD_AGE', 'MAX_LOAN_AGE', 'month_income', 'LOAN_COUNT', 'QUERY_TIMES2']
    index=[1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]


    for m in range(14, len(fea_list)):
        colom = fea_list[m]
        print colom
        print " Continuous :0  not Continuous:1   Continuous  is :",Continuous[m]

        #choose the fig_save_path
        SP = "density"
        fig_save_path='xxd_new_years_density_figure'
        fig_save_path = str(fig_save_path) +'/'
        if not os.path.exists(fig_save_path):
            os.makedirs(fig_save_path)
            print "The path of saving figs has been created"


        try:

            #choose best outlier
            outlier1, outlier2, outlier3 = outliers_detection(new_data[colom])
            print outlier2
            if outlier2>0.0:
                pass
            else:
                outlier2=np.mean([outlier1, outlier2, outlier3])

            #plot density picture
            if Continuous[m] == 0:
                print "it is Continuous"
                density(Dataset1, Dataset2, Dataset3, Dataset4, colom, SP, outlier2, fig_save_path)
            else:
                print "it is not Continuous"
                density(Dataset1, Dataset2, Dataset3, Dataset4, colom, SP, 100, fig_save_path)
        except:
            print "err"


if __name__ == '__main__':
    plot_density()
