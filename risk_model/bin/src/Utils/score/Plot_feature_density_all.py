#-*- coding: UTF-8 -*-
import csv
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
parser = argparse.ArgumentParser()
parser.add_argument('--data_name',type=str,default='xxd_good_and_m7_model',help='training data in csv format')
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

def density(data,col_name,SP,outliers_max,fig_save_path):
    data_notnull = data[-data[col_name].isnull()][col_name]
    data_not_outliers1=data_notnull[data_notnull<=outliers_max]
    sns.distplot(data_not_outliers1, rug=True, hist=False, label='all')
    plt.savefig(str(fig_save_path) + "_" + SP + "_" + str(col_name) +'_single_'+ '.png', dpi=180)
    # plt.show()
    plt.close()



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
    # outerline=["label","id",100,1000000,1000000,100,1000000,1000000,400,100000,3,2,100,10,10,10,10,10,10,10,10,100,10,15,50000,200,10,10,15,800,5000,600,10,10
    #         ,10,10,10,100000,100000,100000,100000,100000,100000,100000,100000,10,10,10,1500,100,100,20,10,0.15,1,5,5,100,50,50,0.5,50,100,10000,500,0.5
    #         ,50000,100,100,0.2,5,2,1,1000,1000,3,3,2,15,15,15,100,100]

    data_name = args.data_name
    new_data = load_data(data_name)
    fea_list = new_data.columns



    for m in range(12, len(fea_list)):
        colom = fea_list[m]
        print colom
        print " Continuous :0  not Continuous:1   Continuous  is :",Continuous[m]

        #choose the fig_save_path
        SP = "density"
        fig_save_path='xxd_new_density_figure'
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
                density(new_data,colom, SP, outlier2, fig_save_path)
            else:
                density(new_data, colom, SP, 100, fig_save_path)
                print "it is not Continuous"

        except:
            print "err"
if __name__ == '__main__':
    plot_density()
