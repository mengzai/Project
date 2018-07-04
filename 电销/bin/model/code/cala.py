# coding:utf-8
import csv

import pandas as pd
import struct


def output_final(data, Path):
    f = open(Path, "w")
    data.to_csv(Path, sep='\t', index=False, header=True)
    f.close()


def data_file_name():
    date_list_10 = ['2016-10-05','2016-10-06','2016-10-07', '2016-10-08',
                    '2016-10-09', '2016-10-10', '2016-10-11', '2016-10-12', '2016-10-13','2016-10-14',
                    '2016-10-15', '2016-10-16', '2016-10-17', '2016-10-18', '2016-10-19', '2016-10-20','2016-10-21', '2016-10-22',
                    '2016-10-23', '2016-10-24', '2016-10-25', '2016-10-26', '2016-10-27', '2016-10-28', '2016-10-29','2016-10-30','2016-10-31']
    date_list_1 = ['2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07','2017-01-08',
                 '2017-01-09','2017-01-10', '2017-01-11', '2017-01-12', '2017-01-13', '2017-01-14', '2017-01-15', '2017-01-16',
                 '2017-01-17','2017-01-18', '2017-01-19', '2017-01-20', '2017-01-21', '2017-01-22', '2017-01-23', '2017-01-24',
                 '2017-01-25','2017-01-26','2017-01-27','2017-01-28','2017-01-29','2017-01-30','2017-01-31']
    return date_list_1, date_list_10


############## 需先按customerid去重 ###########
def dere_id(data):
    data.sort_values(by=['customerid', 'calltime'], ascending=[False, False], inplace=True)
    data['calldate'] = map(lambda x: x[0:10], data['calltime'])
    length = len(data['customerid'])
    data.reset_index(level=range(0, length), inplace=True)
    records = zip(data['customerid'], data['calldate'], data['callhisid'])
    last_cus_id, last_calldate, last_callhisid = records[0]
    del_id = []
    index1 = 1
    for cus_id, calldate, callhisid in records[1:]:
        if cus_id == last_cus_id:
            if calldate == last_calldate:
                del_id.append(index1)
            else:
                last_calldate = calldate
        else:
            last_cus_id = cus_id
            last_calldate = calldate
        index1 += 1
    datanew = data.drop(data.index[del_id])
    return datanew


################## 生成每天的测试数据（生成两组数据，有无customerid） ################
def sub_eachday(datanew, date_list):
    for date in date_list:
        eachday_data = datanew[datanew['calldate'] == date]
        # print eachday_data
        eachday_data = eachday_data[['label', 'weight','call_times', 'connect_times', 'has_callin', 'has_staff_hangup', 'avg_waittime',
                     'min_waittime','max_waittime', 'avg_onlinetime', 'min_onlinetime', 'max_onlinetime', 'datasource', 'callresult',
                     'emotion', 'str_zhengxin','str_jujie', 'str_zhuce', 'str_mingtian', 'str_mendian', 'str_kaolv', 'str_feilv',
                     'str_daka','str_guanji', 'str_bu', 'str_jie', 'gap','mobile_type', 'weekday', 'province', 'onlinetime_gap',
                     'online_ascending_num','online_decsending_num', 'waittime_ascending_num', 'waittime_decsending_num', 'avg_comments_cnt',
                     'one_month_innum','two_month_innum','three_month_innum','beta_online','beta_wait','score','loanamount','sex1','has_car',
                     'house','class','age']]
        # ,'customerid','callhisid','calltime'
        name = str(date) + '_test'

        output_final(eachday_data, '../C++_feature/subdata/' + name)
        # output_final(eachday_data, './subdata_total/' + name)    #加id


######## 拼接最后的结果，返回每天的预测值 ############
def connect_data(date_list):
    last_data = pd.read_csv('./subdata_total/2016-10-05_test', sep='\t', quoting=csv.QUOTE_NONE)
    last_data['predict'] = pd.read_csv('./result_10/2016-10-05_result', sep='\t', quoting=csv.QUOTE_NONE, header=None)
    output_final(last_data[['customerid', 'label', 'predict']], './final_10/2016-10-05')
    last_prob = list(last_data['predict'])
    last_label = list(last_data['label'])
    last_id = list(last_data['customerid'])

    set_id = set(last_id)

    for i in range(1, len(date_list)):
        print date_list[i]
        name1 = str(date_list[i]) + '_test'
        name2 = str(date_list[i]) + '_result'
        cur_data = pd.read_csv('./subdata_total/' + name1, sep='\t', quoting=csv.QUOTE_NONE)
        cur_data['predict'] = pd.read_csv('./result_10/' + name2, sep='\t', quoting=csv.QUOTE_NONE, header=None)
        cur_prob = cur_data['predict']
        cur_label = cur_data['label']
        cur_id = cur_data['customerid']

        for j in range(len(cur_prob)):
            # if cur_id[i] not in prev_id:
            if cur_id[j] not in set_id:
                last_label.append(cur_label[j])
                last_id.append(cur_id[j])
                last_prob.append(cur_prob[j])
                set_id.add(cur_id[j])
            else:
                last_prob[last_id.index(cur_id[j])] = cur_prob[j]

        out = pd.DataFrame()
        out['customerid'] = last_id
        out['label'] = last_label
        out['predict'] = last_prob
        name = str(date_list[i])
        output_final(out, './final_10/' + name)


def main():

    date_list_1, date_list_10 = data_file_name()
    # data_10 = pd.read_csv('../C++_feature/jieqing_test_10_id', sep='\t', quoting=csv.QUOTE_NONE, low_memory=False)
    data_1 = pd.read_csv('../C++_feature/jieqing_test_1_id', sep='\t', quoting=csv.QUOTE_NONE, low_memory=False)
    # print 'done'

    datanew = dere_id(data_1)

    sub_eachday(datanew,date_list_1)

    #拼接只需要这个函数
    # connect_data(date_list_10)


if __name__ == '__main__':
    main()

    # prev_id = list(cur_test['customerid'])
    # prev_label = list(cur_test['label'])
    # value = np.column_stack((prev_label, predict_test))
    # dict_out = dict(zip(prev_id, value))
    # out['customerid'] = list(dict_out.keys())
    # out['test_label'] = np.array(dict_out.values())[:, 0]
    # out['predict_test'] = np.array(dict_out.values())[:, 1]
