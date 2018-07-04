# encoding=utf8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import echarts as ec
import math
import datetime
FORMAT = "%Y-%m-%d"
from matplotlib import pyplot as plt
from collections import Counter


# def process_age(x):
#     if x<=30:
#         return 0
#     elif x<=40:
#         return 1
#     elif x<=50:
#         return 2
#     elif x<=60:
#         return 3
#     else:
#         return 4

def process_recommend_num(x):
    if x>=10:
        return 10
    else:
        return int(x)

def eop_log(x):
	log_eop= int(math.log10(x)) if  x!=0.0 else 0
	return log_eop


def process_province(x):
	x=x.decode('gbk')
	jiangsu_str = "江苏省"
	zhejiang_str = "浙江省"
	beijing_str = "北京市"
	shanghai_str = "上海市"
	shandong_str = "山东省"
	guangdong_str = "广东省"

	if (x!=jiangsu_str) and (x!=zhejiang_str) and (x!=beijing_str) and (x!=shanghai_str) and (x!=shandong_str) and (x!=guangdong_str):
		return 'other'
	else:
		return x


def sex_pivot_table(data,path):
	cus_open_time = pd.pivot_table(data, index=["sex"], columns=["cus_open_time"],aggfunc={ "cus_open_time":len}, fill_value=0)
	age_leval = pd.pivot_table(data, index=["sex"], columns=["age_leval"], aggfunc={"age_leval": len},
								   fill_value=0)

	t_recommend_num = pd.pivot_table(data, index=["sex"], columns=["t_recommend_num"], aggfunc={"t_recommend_num": len},
								   fill_value=0)

	customer_level = pd.pivot_table(data, index=["sex"], columns=["customer_level"], aggfunc={"customer_level": len},
									 fill_value=0)

	group = data.groupby(['sex', 'age_leval'])
	print group.size().unstack().fillna(0)

	customer_level2 = pd.crosstab(data['sex'], data['customer_level'], rownames=['sex'], colnames=['customer_level'])

	t_province = pd.pivot_table(data, index=["sex"], columns=["t_province"], aggfunc={"t_province": len},
	fill_value = 0)

	cus_open_time.to_csv(path+'cus_open_time.csv')
	age_leval.to_csv(path+'age_leval.csv')
	t_recommend_num.to_csv(path+'t_recommend_num.csv')
	customer_level.to_csv(path+'customer_level.csv')
	t_province.to_csv('./data/sf/zh/all/t_province.csv')
	print cus_open_time
	print age_leval
	print t_recommend_num
	print customer_level
	# print customer_level2
	print t_province

def t_recommend_num(data,path):
	t_recommend_num_customer_level = pd.pivot_table(data, index=["t_recommend_num"], columns=["customer_level"], aggfunc={"customer_level": len},
									 fill_value=0)
	t_recommend_num_t_province = pd.pivot_table(data, index=["t_recommend_num"], columns=["t_province"],
									 aggfunc={"t_province": len},
									 fill_value=0)
	print data['t_recommend_num'].value_counts()
	print len(data['t_recommend_num'])

	t_recommend_num_customer_level.to_csv(path + 't_recommend_num_customer_level.csv')
	t_recommend_num_t_province.to_csv(path + 't_recommend_num_t_province.csv')

def df_to_list(df):
    df = df.tolist()
    res = []
    for each in df:
        vals = [k for k in str(each).split("|") if not pd.isnull(k) and k != 'nan' and k!= 'None' and k!="-999" and k!=""]
        res += list(set(vals))
    return res


def  recommend_num_hobby(data):
	print len(data['t_recommend_num'])
	print data['t_recommend_num'].value_counts()
	data1= data[data['t_recommend_num']<3]

	hobby_list = ["fn_hobbies" ]

	# "fn_hobbies"

	print len(data1)
	final_hobby_list=[]
	for hobby_name in hobby_list:
		fn = map(lambda x: x.decode("utf-8"), df_to_list(data1[hobby_name]))
		final_hobby_list.extend(fn)

	final_hobby_dict={}
	for h in list(set(final_hobby_list)):
		if h!='null':
			final_hobby_dict[h]=final_hobby_list.count(h)
	l = sorted(final_hobby_dict.items(), key=lambda d: d[1], reverse=True)
	for i in l:
		print i[1]

def value_zhexian(data):
	data_count = []
	data_list = list(data)
	length=len(data_list)
	data_set = list(set(data_list))
	data_set=sorted(data_set)

	for i in data_set:
		data_count.append(data_list.count(i)*1.0/length)
	return data_set,data_count


def value_zhexian1(data,data_set):
	data_count = []
	data_list = list(data)
	length=len(data_list)

	for i in data_set:
		data_count.append(data_list.count(i)*1.0/length)
	return data_set,data_count
def  find_diff(data):
	data1 = data[data['t_recommend_num'] >= 3]
	data_set, data_count=value_zhexian(data['customer_level'])
	data1_set, data1_count = value_zhexian1(data1['customer_level'],data_set)
	print data_set,data_count
	print data1_set, data1_count

	"""Line 实例"""
	'''折线图的实例'''
	chart = ec.Echart(True, theme='macarons')
	chart.use(ec.Title('MGM 推荐人数>2的客户 与全集客户差异', 'customer_level'))
	chart.use(ec.Tooltip(trigger='axis'))
	chart.use(ec.Legend(data=['all', '推荐人数>2']))
	chart.use(ec.Toolbox(show='true', feature=ec.Feature()))
	chart.use(ec.Axis(param='x', type='category',
				   data=data_set, boundaryGap='false'))
	chart.use(ec.Axis(param='y', type='value'))
	itemStyle = {"normal": {"areaStyle": {"type": 'default'}}}
	chart.use(ec.Line(name='all', data= data_count,itemStyle=itemStyle,
				   smooth='true'))
	chart.use(ec.Line(name='推荐人数>2', data=data1_count,itemStyle=itemStyle,smooth='true'))
	chart.plot()

	"""Pie 实例"""
	'''饼状图的实例'''
	chart = ec.Echart(True, theme='macarons')
	chart.use(ec.Title('MGM 推荐人数>2的客户 与全集客户差异', 'customer_level'))
	chart.use(ec.Tooltip(trigger='axis'))
	chart.use(ec.Legend(data=['all', '推荐人数>2']))
	chart.use(ec.Toolbox(show='true', feature=ec.Feature()))
	chart.use(ec.Axis(param='x', type='category',
					  data=data_set, boundaryGap='false'))
	chart.use(ec.Axis(param='y', type='value'))
	itemStyle = {"normal": {"areaStyle": {"type": 'default'}}}
	chart.use(ec.Line(name='all', data=data_count, itemStyle=itemStyle,
					  smooth='true'))
	chart.use(ec.Line(name='推荐人数>2', data=data1_count, itemStyle=itemStyle, smooth='true'))
	chart.plot()


def process_age(x):
    cutoff_now=datetime.datetime.now()

    cutoff_x = datetime.datetime.strptime(x,FORMAT)
    return int (((cutoff_now-cutoff_x).days)/365)

def process_open(x):
	cutoff_now = datetime.datetime.now()
	x=x.split(" ")[0]
	cutoff_x = datetime.datetime.strptime(x, FORMAT)
	return int(((cutoff_now - cutoff_x).days) / 365)

def density(data1,col_name,fig_save_path):
    data_notnull1 = data1[-data1[col_name].isnull()][col_name]
	#加入范围
    # data_not_outliers1=data_notnull1[data_notnull1<=outliers_max]
    data_not_outliers1=data_notnull1
    sns.distplot(data_not_outliers1, rug=False, hist=False, label=col_name)
    plt.savefig(str(col_name) + '.png', dpi=180)
    plt.show()


def plot_density(colom_name_list,new_data):

    print len(colom_name_list)
    for m in range(0, len(colom_name_list)):
        colom = "t_recommend_num"
        print  colom

        # print " colom   name  is :",colom_name_list[m]

        #choose the fig_save_path
        SP = "density"
        fig_save_path='fig_ori/'
        fig_save_path = str(fig_save_path) + str(colom)

        try:
            density(new_data, colom, fig_save_path)
            break
        except:
            print "err",colom

def  find_activity_cnt(x):
    x1 = x.split('|')
    return len(x1)

def process_activity_num(x):
    if x>=10:
        return 10
    else:
        return int(x)

def main():
    # data = pd.read_csv("./data/sf/zh/MGM_feature_v5.csv")
    data = pd.read_csv("./data/sf/zh/final_version_data.csv")
    data['activity_cnt']=data[['t_activity_name']].apply(lambda x:find_activity_cnt(x[0]),axis=1)
    data['activity_num']=data[['activity_cnt']].apply(lambda x:process_activity_num(x[0]),axis=1)
    print data['activity_num']

    # print data.colmons
	# data['age_leval'] = data[['birth_date']].apply(lambda x: process_age(x[0]), axis=1)
	# data['open_leval'] = data[['open_time']].apply(lambda x: process_open(x[0]), axis=1)

	# data['t_recommend_num'] = data['recommend_num'].apply(lambda x: process_recommend_num(x))

	# plot_density("recommend_num",data)
	# data['t_province'] = data['name'].apply(lambda x: process_province(x))
	# sex_pivot_table(data,'./data/sf/zh/all/')
	# t_recommend_num(data, './data/sf/zh/MGM/')
	# recommend_num_hobby(data)
	#find_diff(data)
if __name__ == '__main__':
	main()