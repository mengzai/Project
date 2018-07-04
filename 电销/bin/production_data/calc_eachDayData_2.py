# coding:utf-8
import csv

import pandas as pd
import datetime
import struct


def data_file_name(month):
    date_list_1 = ['2017-01-03', '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07','2017-01-08',
                 '2017-01-09','2017-01-10', '2017-01-11', '2017-01-12', '2017-01-13', '2017-01-14', '2017-01-15', '2017-01-16',
                 '2017-01-17','2017-01-18', '2017-01-19', '2017-01-20', '2017-01-21', '2017-01-22', '2017-01-23', '2017-01-24',
                 '2017-01-25']
    date_list_2 = ['2017-02-06','2017-02-07','2017-02-08','2017-02-09','2017-02-10','2017-02-11','2017-02-12','2017-02-13',
                   '2017-02-14','2017-02-15','2017-02-16','2017-02-17','2017-02-18','2017-02-19','2017-02-20','2017-02-21',
                   '2017-02-22','2017-02-23','2017-02-24','2017-02-25','2017-02-26','2017-02-27','2017-02-28']
    date_list_3 = ['2017-03-01','2017-03-02','2017-03-03','2017-03-04','2017-03-05','2017-03-06','2017-03-07','2017-03-08',
                   '2017-03-09','2017-03-10','2017-03-11','2017-03-12','2017-03-13','2017-03-14','2017-03-15','2017-03-16',
                   '2017-03-17','2017-03-18','2017-03-19','2017-03-20','2017-03-21','2017-03-22','2017-03-23','2017-03-24',
                   '2017-03-25','2017-03-26','2017-03-27','2017-03-28','2017-03-29','2017-03-30','2017-03-31','2017-04-01',
                   '2017-04-02']
    if month == 1:
        return date_list_1
    if month == 2:
        return date_list_2
    if month == 3:
        return date_list_3


def output_result(data, Path):
    title = ['label','phone','fk_label','call_times','connect_times','has_callin','has_staff_hangup','avg_waittime',
                   'min_waittime','max_waittime','avg_onlinetime','min_onlinetime','max_onlinetime','province','callresult',
                   'str_zhengxin','str_jujie','str_zhuce','str_mingtian','str_mendian','str_kaolv','str_feilv','str_daka','str_guanji',
                   'emotion','weekday','avg_comments_cnt','onlinetime_gap', 'online_ascending_num','online_decsending_num',
                   'waittime_ascending_num', 'waittime_decsending_num','level','month_nums_in','beta_online','beta_wait','score',
                   'loanamount','sex','has_car','house','class','age']
    with open(Path, 'w') as output:
        output.write('%s\n' % '\t'.join(map(lambda x: str(x), title)))
        for item in data:
            output.write('%s\n' % '\t'.join(map(lambda x:str(x),item)))


######### 将字符串转换成datetime类型
def strtodatetime(datestr, format):
    return datetime.datetime.strptime(datestr, format)


######### 计算两个日期间相差的天数
def datediff(beginDate,endDate):
    format="%Y-%m-%d"
    try:
        bd=strtodatetime(beginDate,format)
        ed=strtodatetime(endDate,format)
    except:
        return -1
    oneday=datetime.timedelta(days=1)
    count = 0
    if (bd > ed):
        count = -1
    while bd<=ed:
        ed=ed-oneday
        count+=1
    return count


###### 打标签 #########
def judge_label(phone, calltime, sendtime):
    jinjiandate = str(sendtime)[0:10]
    nowdate = str(calltime)[0:10]
    internal_days = datediff(nowdate, jinjiandate)
    label = -1
    if  (internal_days <= 30) and (internal_days >= 0):
        label = 1
    elif internal_days > 30 or internal_days == -1:
        label = 0
    else:
        print phone
    return label

####### 标记放款 ######
def istime(fktime):
    date = str(fktime)[0:10]
    format = "%Y-%m-%d"
    try:
        strtodatetime(date, format)
        return True
    except:
        return False

def judge_fk(label, fktime):
    if (label == 1) and (istime(fktime) is True):
        fk_label = 1
    else:
        fk_label = 0
    return fk_label


################## 将的测试数据拆成每天（1.3,2.6,3.1全量数据，根据每月第一天生成后续，标签动态） ################
def sub_eachday(date_list, month):
    callhisid_dict = {}

    ######## read data ######
    print 'reading...'
    data = pd.read_csv("../C++_feature/jieqing_%s"%month, sep='\t', quoting=csv.QUOTE_NONE, low_memory=False, iterator=True)
    loop = True
    chunkSize = 100000
    chunks = []
    while loop:
        try:
            chunk = data.get_chunk(chunkSize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print "Iteration is stopped."

    data = pd.concat(chunks, ignore_index=True)
    print 'done'
    ######## end ###########

    data['calldate'] = map(lambda x: x[0:10], data['calltime'])
    start_day = date_list[0]
    start_day_data = data[data['calldate'] == start_day]
    start_day_data['label'] = [0]*len(start_day_data)
    start_day_data['fk_label'] = [0]*len(start_day_data)
    for i in range(len(start_day_data)):
        line = start_day_data.iloc[i]
        line['label'] = judge_label(line['phone'], line['calltime'], line['sendtime'])  ##修改标签
        line['fk_label'] = judge_fk(line['label'], line['back_time'])
        line = line[['label','phone','fk_label','call_times','connect_times','has_callin','has_staff_hangup','avg_waittime',
                   'min_waittime','max_waittime','avg_onlinetime','min_onlinetime','max_onlinetime','province','callresult',
                   'str_zhengxin','str_jujie','str_zhuce','str_mingtian','str_mendian','str_kaolv','str_feilv','str_daka','str_guanji',
                   'emotion','weekday','avg_comments_cnt','onlinetime_gap', 'online_ascending_num','online_decsending_num',
                   'waittime_ascending_num', 'waittime_decsending_num','level','month_nums_in','beta_online','beta_wait','score',
                   'loanamount','sex','has_car','house','class','age']]
        callhisid_dict[line['phone']] = line.values  ## 记录当天的拨打过的phone数据

    ## 输出第一天 ##
    one_final_data = []
    for phone in callhisid_dict:
        line_data = callhisid_dict[phone].tolist()
        one_final_data.append(line_data)

    output_result(one_final_data, './final_%s/%s' % (month,start_day))

    ## 循环处理余下n-1天 ##
    for i in range(1, len(date_list)):
        print date_list[i]
        now_day = date_list[i]
        now_day_data = data[data['calldate'] == now_day]
        now_day_data['label'] = [0] * len(now_day_data)
        now_day_data['fk_label'] = [0] * len(now_day_data)
        for i in range(len(now_day_data)):
            line = now_day_data.iloc[i]
            line['label'] = judge_label(line['phone'], line['calltime'], line['sendtime'])
            line['fk_label'] = judge_fk(line['label'], line['back_time'])
            line = line[['label','phone','fk_label','call_times','connect_times','has_callin','has_staff_hangup','avg_waittime',
                   'min_waittime','max_waittime','avg_onlinetime','min_onlinetime','max_onlinetime','province','callresult',
                   'str_zhengxin','str_jujie','str_zhuce','str_mingtian','str_mendian','str_kaolv','str_feilv','str_daka','str_guanji',
                   'emotion','weekday','avg_comments_cnt','onlinetime_gap', 'online_ascending_num','online_decsending_num',
                   'waittime_ascending_num', 'waittime_decsending_num','level','month_nums_in','beta_online','beta_wait','score',
                   'loanamount','sex','has_car','house','class','age']]
            callhisid_dict[line['phone']] = line.values  ##更新或新增数据字典

        ## 输出n-1天 ##
        final_data = []
        for phone in callhisid_dict:
            line_data = callhisid_dict[phone].tolist()
            final_data.append(line_data)

        output_result(final_data, './final_%s/%s' % (month,now_day))



def main():
    month = 1
    date_list = data_file_name(month)  ###  在这里改月份
    sub_eachday(date_list,month)


if __name__ == '__main__':
    main()
