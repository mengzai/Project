# coding=utf8=

import sys
import matplotlib.pyplot as plt
import pandas as pd
from snownlp import SnowNLP
import numpy as np
import datetime
import csv
import re
import jieba
import jieba.analyse

import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

jieba.load_userdict('data/user_dict')

try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

prov_dict = {"北京": 1, "天津": 2, "上海": 3, "重庆": 4, "河北": 5, "河南": 6, "云南": 7, "辽宁": 8, "黑龙江": 9, "湖南": 10, "安徽": 11,
             "山东": 12, "新疆": 13,
             "江苏": 14, "浙江": 15, "江西": 16, "湖北": 17, "广西": 18, "甘肃": 19, "山西": 20, "内蒙古": 21, "陕西": 22, "吉林": 23,
             "福建": 24, "贵州": 25,
             "广东": 26, "青海": 27, "西藏": 28, "四川": 29, "宁夏": 30, "海南": 31, "台湾": 32, "香港": 33, "澳门": 34, "other": 35}


########## 把所有词和双词搭配一起作为特征 #############
def bag_of_words(data):
    lwords = fenci(data)
    words = lwords
    return dict([(word, True) for word in words])


def bigram_words(data, score_fn=BigramAssocMeasures.chi_sq, n=1000):
    lwords = fenci(data)
    words = lwords
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(words + bigrams)  # 所有词和（信息量大的）双词搭配一起作为特征


###### 双词 #######
def bigram(lwords, score_fn=BigramAssocMeasures.chi_sq, n=2):
    # lwords = fenci(data)
    bigram_finder = BigramCollocationFinder.from_words(lwords)  # 把文本变成双词搭配的形式
    bigrams = bigram_finder.nbest(score_fn, n)  # 使用了卡方统计的方法，选择排名前1000的双词
    return bigrams


######## 三词搭配 #######
# def trigrams(words,score_fn=BigramAssocMeasures.chi_sq, n=1000):
#     bigram_finder = BigramCollocationFinder.from_words(words)
#     bigrams = bigram_finder.nbest(score_fn, n)
#     return bag_of_words(words + bigrams)  # 所有词和（信息量大的）双词搭配一起作为特征


def output_result(value, output_file_name):
    with open(output_file_name, 'w') as output:
        i = 0
        for item in value:
            output.write('%s\t\t\t' % ('--').join(map(lambda x: repr(x).decode('unicode-escape'), item)))
            if (i % 20 == 0):
                output.write('\n')
            i += 1


def output_final(data, Path):
    f = open(Path, "w")
    data.to_csv(Path, sep='\t', index=False, header=True)
    f.close()


def process_location(data, prov_dict):
    location = list(data['location'])
    location_tmp = []
    i = 1
    for item in location:
        if (str(item) == "nan"):
            location_tmp.append(None)
            continue
        prov = item.split(" ")
        location_tmp.append(prov[0])
        if (not prov_dict.has_key(prov[0])):
            prov_dict[prov[0]] = i
            i = i + 1

    location_new = []
    for item in location_tmp:
        if (item is None):
            location_new.append(None)
            continue
        location_new.append(prov_dict[item])

    data['province'] = location_new
    return data


def calc_weight(data):
    id = list(data['customerid'])
    id_dict = {}
    for item in id:
        if (not id_dict.has_key(item)):
            id_dict[item] = 1
        else:
            id_dict[item] = id_dict[item] + 1

    weight = [1.0 / id_dict[id[i]] for i in range(len(id))]
    return pd.Series(weight)


def calc_weight_mul(data, w):
    id = list(data['customerid'])
    label = list(data['label'])
    id_dict = {}
    for item in id:
        if (not id_dict.has_key(item)):
            id_dict[item] = 1
        else:
            id_dict[item] = id_dict[item] + 1

    weight = map(lambda x: 1.0 / id_dict[id[x]] if label[x] == 0 else  w * 1.0 / id_dict[id[x]], range(len(id)))
    return pd.Series(weight)


def fenci(data):
    comments = list(data['comments'])
    regex = re.compile("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）;；-]+".decode("utf8"))
    # regex2 = re.compile('[%s]' % re.escape(string.punctuation))
    result = []
    for item in comments:
        temp = str(item)
        temp = temp.strip('\n')
        temp = regex.sub('', temp)
        temp = temp.replace('。', '')
        temp = temp.strip()
        tags = jieba.lcut(temp, cut_all=False)
        tags2 = del_num_in_item(tags)
        result.append(bigram(tags2))
    # tags = jieba.analyse.extract_tags(full_text)    #, topK=n, withWeight= True
    return result


def fenci2(data):
    comments = list(data['comments'])
    cut_text = []
    for item in comments:
        temp = str(item)
        tags = jieba.lcut(temp, cut_all=False)
        tags2 = del_num_in_item(tags)
        cut_text.append(tags2)
    return cut_text


def hasNumbers(inputString):
    if str(inputString) == 'nan' or bool(re.search('.*([0-9]+).*', inputString)):
        return True


def del_num_in_item(sublist):
    for i in range(len(sublist) - 1, -1, -1):
        if hasNumbers(sublist[i]) == True:
            sublist.remove(sublist[i])
    return sublist


def qinggan(data):
    comments = list(data['comments'])
    emotion = []
    for item in comments:
        temp = str(item)
        word = SnowNLP(temp)
        emotion.append(word.sentiments)
    return emotion


def create_str(data, str_name):
    str_col = list(data['comments'])
    str_col_new = map(lambda x: 1 if str(x) != 'nan' and x.find(str_name) >= 0 else 0, str_col)
    return pd.Series(str_col_new)


def cal_weekday(data):
    call_data = list(data['calltime'])
    day = []
    for item in call_data:
        date = datetime.datetime.strptime(item, "%Y-%m-%d %H:%M:%S")
        day.append(date.weekday())
    day = pd.Series(day)
    return day


def cal_percent(data, feature_name):
    subdata = data[feature_name]
    zero = 0
    one = 0
    total = len(subdata)
    print 'total:'+str(total)
    for i in range(0, len(subdata)):
        if subdata[i] == 0:
            zero += 1
        if subdata[i] == 1:
            one += 1
    per_zero = float(zero * 1.0 / total)
    per_one = float(one * 1.0 / total)
    print 'zero:' + str(zero) + '\n' + 'one:' + str(one) + '\n' + 'per_zero:' + str(per_zero) + '\n' + 'per_one:' + str(
        per_one)


def cal_percent2(data, feature_name1, festure_name2):
    subdata1 = data[feature_name1]
    subdata2 = data[festure_name2]
    setzero = set()
    setone = set()
    zero = 0
    one = 0
    # print 'total:'+str(total)
    for i in range(0, len(subdata1)):
        if subdata1[i] == 0:
            if (subdata2[i] in setzero) == False:
                zero += 1
                setzero.add(subdata2[i])
        if subdata1[i] == 1:
            if (subdata2[i] in setone) == False:
                one += 1
                setone.add(subdata2[i])
    total2 = len(setzero) + len(setone)
    print 'total:' + str(total2)
    per_zero = float(zero * 1.0 / total2)
    per_one = float(one * 1.0 / total2)
    print 'zero:' + str(zero) + '\n' + 'one:' + str(one) + '\n' + 'per_zero:' + str(per_zero) + '\n' + 'per_one:' + str(
        per_one)


def calc_ascending_count(data_column):
    asc_count_list = []
    for person_tel_records in data_column:
        tel_records = sorted(set(person_tel_records.split(',')), reverse=True)
        asc_count_list.append(cal_acs_tel_records(tel_records))
    return asc_count_list


def cal_acs_tel_records(tel_records):
    last_tel_duration = 2 << 32
    ascending_count = -1
    for tel_record in tel_records:
        tel_time, tel_duration = tel_record.split('@')
        if tel_duration == '':
            continue
        tel_duration = int(tel_duration)
        if tel_duration == 0:
            return 0
        if last_tel_duration < tel_duration:
            return ascending_count
        ascending_count += 1
        last_tel_duration = tel_duration
    return ascending_count


def calc_decsending_count(data_column):
    asc_count_list = []
    for person_tel_records in data_column:
        tel_records = sorted(set(person_tel_records.split(',')), reverse=True)
        asc_count_list.append(cal_decs_tel_records(tel_records))
    return asc_count_list


def cal_decs_tel_records(tel_records):
    last_tel_duration = 0
    descending_count = -1
    for tel_record in tel_records:
        tel_time, tel_duration = tel_record.split('@')
        if tel_duration == '':
            continue
        tel_duration = int(tel_duration)
        if tel_duration > 10000:
            return 0
        if last_tel_duration > tel_duration:
            return descending_count
        descending_count += 1
        last_tel_duration = tel_duration
    return descending_count



def comments_cnt(data):
    comments = list(data['comments'])
    times = list(data['call_times'])
    cnt = map(lambda x, y: None if str(x) == 'nan' else len(re.sub(r'[0-9, ;.\n-]', '', x).decode('utf-8')) * 1.0 / y,
              comments, times)
    return cnt


def calc_beta(data_column):
    beta_list = []
    for person_tel_records in data_column:
        tel_records = sorted(set(person_tel_records.split(',')), reverse=True)
        tel_duration = map(lambda x: x.split('@')[1], tel_records)
        tel_duration = [0 if x=='' else float(x) for x in tel_duration]
        if len(tel_duration)>1:
            res = np.polyfit(range(1, len(tel_duration) + 1), tel_duration, 1)
            beta_list.append(res[0])
        else:
            beta_list.append(None)
    return beta_list


def main():

    ########### read data ################
    data = pd.read_csv("../total_data/dianxiao_10_qk", sep='\t', quoting=csv.QUOTE_NONE, low_memory=False, iterator=True)
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


    ########### temp output #########
    # cal_percent2(data,'label','customerid')


    ########### process data ##############3
    data = process_location(data, prov_dict)

    # data = data[data['max_onlinetime'] <= 10000]
    # data = data[data['min_onlinetime'] <= 10000]

    weight = calc_weight(data)
    emotion = qinggan(data)
    day = cal_weekday(data)
    data['weekday'] = day
    data['str_zhengxin'] = create_str(data, '征信')
    data['str_jujie'] = create_str(data, '拒接')
    data['str_zhuce'] = create_str(data, '注册')
    data['str_mingtian'] = create_str(data, '明天')
    data['str_mendian'] = create_str(data, '门店')
    data['str_kaolv'] = create_str(data, '考虑')
    data['str_feilv'] = create_str(data, '费率')
    data['str_daka'] = create_str(data, '打卡')
    data['str_guanji'] = create_str(data, '关机')
    data['str_bu'] = create_str(data, '不')
    data['str_jie'] = create_str(data, '接')
    #
    data['onlinetime_gap'] = abs(data['max_onlinetime'] - data['min_onlinetime'])
    data['online_ascending_num'] = calc_ascending_count(data['all_call_onlinetime'])
    data['online_decsending_num'] = calc_decsending_count(data['all_call_onlinetime'])
    data['waittime_ascending_num'] = calc_ascending_count(data['all_call_waittime'])
    data['waittime_decsending_num'] = calc_decsending_count(data['all_call_waittime'])
    data['emotion'] = emotion
    data['avg_comments_cnt'] = comments_cnt(data)   #按customerid去重
    data['beta_online'] = calc_beta(data['all_call_onlinetime'])
    data['beta_wait'] = calc_beta(data['all_call_waittime'])

    data_new = data[['label','call_times', 'connect_times', 'has_callin', 'has_staff_hangup', 'avg_waittime',
                     'min_waittime','max_waittime', 'avg_onlinetime', 'min_onlinetime', 'max_onlinetime', 'datasource', 'callresult',
                     'emotion', 'str_zhengxin','str_jujie', 'str_zhuce', 'str_mingtian', 'str_mendian', 'str_kaolv', 'str_feilv',
                     'str_daka','str_guanji', 'str_bu', 'str_jie', 'is_old', 'gap','mobile_type', 'weekday', 'province', 'onlinetime_gap',
                     'online_ascending_num','online_decsending_num', 'waittime_ascending_num', 'waittime_decsending_num', 'avg_comments_cnt',
                     'one_month_innum','two_month_innum','three_month_innum','beta_online','beta_wait','customerid','callhisid','calltime']]


    output_final(data_new, '../C++_feature/data_new_10')

if __name__ == '__main__':
    main()
