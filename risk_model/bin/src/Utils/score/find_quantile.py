#encoding=utf-8

import csv
import scipy.stats as stats
import argparse
import pandas as pd
import time
import datetime
parser = argparse.ArgumentParser()
parser.add_argument('--data_name',type=str,default='data_all/xxd_good_and_m7',help='training data in csv format')
parser.add_argument('--logistic_regression',type=str,default='logistic_regression',help='training data in csv format')
parser.add_argument('--term',type=str,default=24,help='training data in csv format')
args = parser.parse_args()


def load_data(path):
    return pd.read_csv(path, error_bad_lines=False)
def des(data,colom,persent):
    decrib = ['feature', 'count（not nullall_count:761539', 'min', 'max', 'mean', '方差', '偏度', '峰度',
              '25%', '50%', '75%', '90%', '99.7%', '99.97%所占人数', '覆盖率', '划分（针对离散型变量）"	含义	"连续1:离散：0', 'meaning']
    savename=args.savename
    file0 = open(savename, 'wb+')  # 'wb'
    output = csv.writer(file0, dialect='excel')
    output.writerow(decrib)
    total = len(data)
    fea_list = data.columns


    data_notnull = data[-data[colom].isnull()][colom]
    print len(data_notnull)
    g_dist = sorted(data_notnull)
    lenth = len(g_dist)
    info = stats.describe(data_notnull)
    # listdes = [colom, str(info[0]), str(info[1][0]), str(info[1][1]), str(info[2]),
    #            str(info[3]), str(info[4]), str(info[5]), g_dist[int(0.25 * lenth)],
    #            g_dist[int(0.5 * lenth)], g_dist[int(0.75 * lenth)], g_dist[int(0.9 * lenth)],
    #            g_dist[int(0.9997 * lenth)], int(lenth - int(0.9997 * lenth)), float(int(info[0]) * 1.0 / total)]
    print g_dist[int(persent* lenth)]


def spit_term_dataframe(dataframe_my,term):

    start_data = '2013-05-01 00:00:00'
    end_data = '2014-10-01 00:00:00'
    spilt_data='2014-05-01 00:00:00'
    start_data = datetime.datetime.strptime(start_data, '%Y-%m-%d %H:%M:%S')
    end_data = datetime.datetime.strptime(end_data, '%Y-%m-%d %H:%M:%S')

    print len(dataframe_my)

    dataframe_my=dataframe_my[dataframe_my['loan_term']==term]

    dataframe_my = dataframe_my[dataframe_my["issue_date"] >= str(start_data)]
    dataframe_my = dataframe_my[dataframe_my["issue_date"] <str(end_data)]

    traindata=dataframe_my[dataframe_my["issue_date"] < str(spilt_data)]
    testdata = dataframe_my[dataframe_my["issue_date"] >= str(spilt_data)]
    print len(dataframe_my),len(traindata),len(testdata)
    return dataframe_my,traindata,testdata


if __name__ == '__main__':
    dataname=args.data_name
    data = load_data(dataname)
    term=args.term
    data_feature_name2 = ['org_type', 'GENDER', "card_interrupt", "HOUSE_CONDITION", "month_income", "n4", "mean", "n2",
                          "in_city_years", \
                          "credit_level", "MAX_CREDIT_CARD_AGE", "CREDIT_CARD_MAX_NUM", "latest_month_income",
                          "QUERY_TIMES2", "gap", "QUERY_TIMES9", "JOB_POSITION", "MAX_LOAN_AGE", "INDUSTRY1",
                          "LONG_REPAYMENT_TERM",
                          "age", "MAX_CREDIT_LINE", "credit_grade", 'label', "APPLY_MAX_AMOUNT", 'label_profit',
                          'profit', "issue_date",
                          "loan_term"]

    dataframe_my = data[data_feature_name2]
    dataframe_my, traindata, testdata = spit_term_dataframe(dataframe_my, term)


    print len(dataframe_my),len(traindata),len(testdata)
    des(data,"GENDER",0.)