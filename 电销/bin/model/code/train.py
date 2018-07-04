# coding=utf8=
import datetime
import re
import sys

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

def process_location(location, prov_dict):

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

    return location_new

prov_dict = {"北京": 1, "天津": 2, "上海": 3, "重庆": 4, "河北": 5, "河南": 6, "云南": 7, "辽宁": 8, "黑龙江": 9, "湖南": 10, "安徽": 11,
             "山东": 12, "新疆": 13,
             "江苏": 14, "浙江": 15, "江西": 16, "湖北": 17, "广西": 18, "甘肃": 19, "山西": 20, "内蒙古": 21, "陕西": 22, "吉林": 23,
             "福建": 24, "贵州": 25,
             "广东": 26, "青海": 27, "西藏": 28, "四川": 29, "宁夏": 30, "海南": 31, "台湾": 32, "香港": 33, "澳门": 34, "other": 35}


def calc_weight(id):
    id_dict = {}
    for item in id:
        if (not id_dict.has_key(item)):
            id_dict[item] = 1
        else:
            id_dict[item] = id_dict[item] + 1

    weight = [1.0 / id_dict[id[i]] for i in range(len(id))]
    return weight


def cal_weekday(call_data):
    day = []
    for item in call_data:
        date = datetime.datetime.strptime(item, "%Y-%m-%d %H:%M:%S")
        day.append(date.weekday())
    return day


def create_str(str_col, str_name):
    str_col_new = map(lambda x: 1 if str(x) != 'nan' and x.find(str_name) >= 0 else 0, str_col)
    return str_col_new


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

def comments_cnt(comments,times):
    cnt = map(lambda x, y: None if str(x) == 'nan' else len(re.sub(r'[0-9, ;.\n-]', '', x).decode('utf-8')) * 1.0 / float(y),
              comments, times)
    return cnt
def main():
	location=[]
	customerid=[]
	calltime=[]
	comments=[]
	onlinetime_gap=[]
	all_call_onlinetime=[]
	all_call_waittime=[]
	call_times=[]

	data=[]
	for line in sys.stdin:
		li = line.rstrip("\n").split("\t")
		data.append(li)

		location.append(li[17])
		customerid.append(li[1])
		calltime.append(li[19])
		comments.append(li[15])
		if li[12] in  ["", '\\N', 'NULL', 'null']:
			onlinetime_gap.append("NULL")
		else:
			onlinetime_gap.append(abs(float(li[12])-float(li[11])))
		all_call_onlinetime.append(li[14])
		all_call_waittime.append(li[13])
		call_times.append(li[3])

	location_new=process_location(location, prov_dict)
	weight = calc_weight(customerid)
	day=cal_weekday(calltime)
	str_zhengxin=create_str(comments, '征信')
	str_jujie = create_str(comments, '拒接')
	str_zhuce = create_str(comments, '注册')
	str_mingtian = create_str(comments, '明天')
	str_mendian = create_str(comments, '门店')
	str_kaolv = create_str(comments, '考虑')
	str_feilv = create_str(comments, '费率')
	str_daka = create_str(comments, '打卡')
	str_guanji = create_str(comments, '关机')
	str_bu = create_str(comments, '不')
	str_jie = create_str(comments, '接')

	online_ascending_num = calc_ascending_count(all_call_onlinetime)
	online_decsending_num = calc_decsending_count(all_call_onlinetime)
	waittime_ascending_num = calc_ascending_count(all_call_waittime)
	waittime_decsending_num = calc_decsending_count(all_call_waittime)
	avg_comments_cnt = comments_cnt(comments,call_times)

	for i in range(len(location_new)):
		write_row = [data[i][0],weight[i],data[i][0][1:],day[i],str_zhengxin[i],str_jujie[i],str_zhuce[i],
					 str_mingtian[i],str_mendian[i],str_kaolv[i],str_feilv[i],str_daka[i],str_guanji[i],
					 str_bu[i],str_jie[i],online_ascending_num[i],online_decsending_num[i],waittime_ascending_num[i],
					 waittime_decsending_num[i],avg_comments_cnt[i]]
		sys.stdout.write("%s\n" % ("\t".join(write_row)))


if __name__ == '__main__':
    main()
