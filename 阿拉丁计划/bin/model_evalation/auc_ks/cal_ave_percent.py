# coding:utf-8
import csv

import pandas as pd
import struct
import datetime

def dateRange(start, end, step=1, format="%Y-%m-%d"):
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    days = (strptime(end, format) - strptime(start, format)).days
    return [strftime(strptime(start, format) + datetime.timedelta(i), format) for i in xrange(0, days, step)]


def data_file_name(month):
    if month == 10:
        date_list_10 = dateRange("2016-10-08", "2016-11-01")
        return date_list_10
    if month == 11:
        date_list_11 = dateRange("2016-11-01", "2016-12-01")
        return date_list_11
    if month == 12:
        date_list_12 = dateRange("2016-12-01", "2017-01-03")
        return date_list_12
    if month == 2:
        date_list_2 = dateRange('2017-02-06', '2017-03-01')
        return date_list_2
    if month == 3:
        date_list_3 = dateRange('2017-03-01', '2017-04-03')
        return date_list_3
    if month == 4:
        date_list_4 = dateRange('2017-04-03', '2017-05-02')
        return date_list_4


def output_final(data, Path):
    f = open(Path, "w")
    data.to_csv(Path, sep='\t', index=False, header=True)
    f.close()


######## 将全量数据拆分为每天的 ############
def connect_data(date_list, month, outpath):
    totaldata = pd.read_csv('../total_data/'+month, sep='\t', quoting=csv.QUOTE_NONE, low_memory=False)
    totaldata['calldate'] = map(lambda x: x[0:10], totaldata['calltime'])

    for i in range(len(date_list)):
        name = str(date_list[i])
        print name
        subdata = totaldata[totaldata['calldate'] == name]
        output_final(subdata, './tongji/%s/%s'%(outpath, name))


def main():
    date_list_10, date_list_11, date_list_12, date_list_1,date_list_2= data_file_name()
    # connect_data(date_list_10, 'calc_10_v3','submonth_10')
    # connect_data(date_list_11, 'calc_11_v3','submonth_11')
    # connect_data(date_list_12, 'calc_12_v3','submonth_12')
    # connect_data(date_list_1, 'calc_1_v3','submonth_1')
    connect_data(date_list_12, 'calc_12_v2', 'submonth_12')

if __name__ == '__main__':
    main()

    # prev_id = list(cur_test['customerid'])
    # prev_label = list(cur_test['label'])
    # value = np.column_stack((prev_label, predict_test))
    # dict_out = dict(zip(prev_id, value))
    # out['customerid'] = list(dict_out.keys())
    # out['test_label'] = np.array(dict_out.values())[:, 0]
    # out['predict_test'] = np.array(dict_out.values())[:, 1]
