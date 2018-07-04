#-*- coding: UTF-8 -*-
import csv
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import datetime
from scipy import stats
from numpy.random import randn
parser = argparse.ArgumentParser()
parser.add_argument('--data_name',type=str,default='data_all/xxd_good_and_m7',help='training data in csv format')
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

def snsplot(data):
    length=len(data)
    xx = np.linspace(-4, 4, length)
    bandwidth = ((4 * data.std() ** 5) / (3 * len(data))) ** .2
    bandwidth = len(data) ** (-1. / 5)

    kernels = []
    for d in data:
        kernel = stats.norm(d, bandwidth).pdf(xx)
        kernels.append(kernel)
        # Scale for plotting
        kernel /= kernel.max()
        kernel *= .4
    f, (ax1) = plt.subplots(1, 1, sharex=True)

    # color_palette 就是要画图用的 "调色盘"
    c1, c2 = sns.color_palette("husl", 3)[:2]
    summed_kde = np.sum(kernels, axis=0)/length
    ax1.plot(xx, summed_kde, c=c1)
    sns.rugplot(data, c=c1, ax=ax1)
    plt.show()

def density(data1,data2,data3,data4,data5,data6,col_name,SP,outliers_max,fig_save_path):
    data1=pd.DataFrame(data1)
    data2 = pd.DataFrame(data2)
    data3 = pd.DataFrame(data3)
    data4 = pd.DataFrame(data4)
    data5 = pd.DataFrame(data5)
    data6 = pd.DataFrame(data6)

    data_notnull1 = data1[-data1[col_name].isnull()][col_name]
    data_notnull2 = data2[-data2[col_name].isnull()][col_name]
    data_notnull3 = data3[-data3[col_name].isnull()][col_name]
    data_notnull4 = data4[-data4[col_name].isnull()][col_name]
    data_notnull5 = data5[-data5[col_name].isnull()][col_name]
    data_notnull6 = data6[-data6[col_name].isnull()][col_name]

    data_not_outliers1=data_notnull1[data_notnull1<=outliers_max]
    data_not_outliers2 = data_notnull2[data_notnull2<=outliers_max]
    data_not_outliers3 = data_notnull3[data_notnull3<=outliers_max]
    data_not_outliers4 = data_notnull4[data_notnull4<=outliers_max]
    data_not_outliers5 = data_notnull5[data_notnull5 <= outliers_max]
    data_not_outliers6 = data_notnull6[data_notnull6 <= outliers_max]


    sns.distplot(data_not_outliers1/len(data_not_outliers1), rug=True, hist=False, label='13_4',color = "b")
    sns.distplot(data_not_outliers2/len(data_not_outliers2), rug=True, hist=False, label='13_7',color = "r")
    sns.distplot(data_not_outliers3/len(data_not_outliers3), rug=True, hist=False, label='13_10',color = "y")
    sns.distplot(data_not_outliers4/len(data_not_outliers4), rug=True, hist=False, label='14_1',color = "g")
    sns.distplot(data_not_outliers5/len(data_not_outliers5), rug=True, hist=False, label='14_4', color="c")
    sns.distplot(data_not_outliers6/len(data_not_outliers6), rug=True, hist=False, label='14_7', color="m")

    # snsplot(data_not_outliers1)
    # snsplot(data_not_outliers2)
    # snsplot(data_not_outliers3)
    # snsplot(data_not_outliers4)
    # snsplot(data_not_outliers5)
    # snsplot(data_not_outliers6)
    plt.savefig(str(fig_save_path) + "_" + SP + "_" + str(col_name) +'_single_'+ '.png', dpi=180)
    plt.close()


def load_data(path):
    return pd.read_csv(path, error_bad_lines=False)


def plot_density():

    #连续与否:  0:连续   1:不连续
    Continuous = ["b.id_number", "a.apply_id", "b.transport_id", "b.mortgagor_id", "a.contract_no", "mm",
                  "a.issue_date", "a.m7_ratio", "a.revenue", "a.total_expense",
                  "a.profit", "a.label", "a.label_profit", 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0,
                  1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0]

    data_name = args.data_name
    new_data = load_data(data_name)
    fea_list = new_data.columns

    monthdays = 90
    start_data = '2013-04-01 00:00:00'
    end_data = '2014-07-01 00:00:00'
    start_data = datetime.datetime.strptime(start_data, '%Y-%m-%d %H:%M:%S')
    end_data = datetime.datetime.strptime(end_data, '%Y-%m-%d %H:%M:%S')
    windows = (end_data - start_data).days / monthdays
    start = '2013-04-01 00:00:00'
    end = '2014-04-01 00:00:00'
    start = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    end = datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S')

    test_feature=['GENDER', 'grade_version','HIGHEST_DIPLOMA','no_interrupted_card_num','card_interrupt', 'loan_purpose', 'n3',
                  'LONG_REPAYMENT_TERM', 'age', 'APPLY_MAX_AMOUNT','ACCEPT_MOTH_REPAY', 'credit_grade',  'in_city_years',
                   'MAX_CREDIT_CARD_AGE', 'MAX_LOAN_AGE', 'month_income', 'LOAN_COUNT', 'QUERY_TIMES2']
    index=[1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    Dataset_list=[]
    for m in range(13, len(fea_list)):
        colom = fea_list[m]
        print colom
        print " Continuous :0  not Continuous:1   Continuous  is :",index[m]

        #choose the fig_save_path
        SP = "density"
        fig_save_path='xxd_density_windows'
        fig_save_path = str(fig_save_path) +'/'


        if not os.path.exists(fig_save_path):
            os.makedirs(fig_save_path)
            print "The path of saving figs has been created"

        for i in range(windows + 1):
            Dataset = new_data[new_data["issue_date"] >= str(start)]
            Dataset = Dataset[Dataset["issue_date"] <= str(end)]
            Dataset_list.append(Dataset)
            start = start + datetime.timedelta(days=monthdays)
            end = end + datetime.timedelta(days=monthdays)

        #choose best outlier
        print len(Dataset_list[1]),len(Dataset_list[2]),len(Dataset_list[3]),len(Dataset_list[4]),len(Dataset_list[5])
        outlier1, outlier2, outlier3 = outliers_detection(new_data[colom])
        print outlier2
        if outlier2>0.0:
            pass
        else:
            outlier2=np.mean([outlier1, outlier2, outlier3])

        #plot density picture
        if Continuous[m] == 0:
            print "it is Continuous"
            density(Dataset_list[0], Dataset_list[1], Dataset_list[2], Dataset_list[3], Dataset_list[4],Dataset_list[5],colom, SP, outlier2, fig_save_path)
        else:
            print "it is not Continuous"
            # density(Dataset_list[0], Dataset_list[1], Dataset_list[2], Dataset_list[3], Dataset_list[4],Dataset_list[5], colom, SP, 100, fig_save_path)
if __name__ == '__main__':
    plot_density()
