# encoding=utf8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import pandas as pd
import echarts as ec
FORMAT = "%Y-%m-%d"


def value_zhexian(data):
	data_count = []
	data_list = list(data)
	length=len(data_list)
	data_set = list(set(data_list))
	data_set=sorted(data_set)

	for i in data_set:
		data_count.append("%.4f" % (data_list.count(i) * 1.0 / length))
	return data_set,data_count

def value_zhexian1(data,data_set):
	data_count = []
	data_list = list(data)
	length=len(data_list)

	for i in data_set:
		data_count.append("%.4f" % (data_list.count(i) * 1.0 / length))
	return data_set,data_count

def  plot_des(data_col,all_data_col):
	data_col0=data_col[data_col['recommend_num'] < 4]
	data_col1 = data_col[data_col['recommend_num'] >=4]
	print len(data_col0),len(data_col1)

	col_list=['loyalty_days']
	for col in col_list:
		all_data_col.loc[(all_data_col[col].isnull()), col] = '-999'
		data_col0.loc[(data_col0[col].isnull()), col] = '-999'
		data_col1.loc[(data_col1[col].isnull()), col] = '-999'

		all_data_col.loc[(all_data_col[col] == -10), col] = '-999'
		data_col0.loc[(data_col0[col] == -10), col] = '-999'
		data_col1.loc[(data_col1[col] == -10), col] = '-999'

		data_set, data_count = value_zhexian(all_data_col[col])
		data1_set, data1_count = value_zhexian1(data_col0[col], data_set)
		data2_set, data2_count = value_zhexian1(data_col1[col], data_set)
		print data_set, data_count
		print data1_set, data1_count
		print data2_set, data2_count

		"""柱状图实例"""
		chart = ec.Echart(True, theme='macarons')
		itemStyle = ec.ItemStyle(normal={'label': {'show': 'true', 'position': 'top', 'formatter': '{c}'}})

		chart.use(ec.Title('全量数据与MGM 数据 直方图',col))
		chart.use(ec.Tooltip(trigger='axis'))
		chart.use(ec.Legend(data=['new_data_all', '推荐人数<4','推荐人数>=4']))
		chart.use(ec.Toolbox(show='true', feature=ec.Feature()))
		chart.use(ec.Axis(param='x', type='category',
		                  data=data_set))
		chart.use(ec.Axis(param='y', type='value'))
		chart.use(ec.Bar(name='new_data_all', data=data_count,itemStyle=itemStyle,
		                 markPoint=ec.MarkPoint(), markLine=ec.MarkLine()))
		chart.use(ec.Bar(name='推荐人数<4', data=data1_count,itemStyle=itemStyle,
		                 markLine=ec.MarkLine()))
		chart.use(ec.Bar(name='推荐人数>=4', data=data2_count,itemStyle=itemStyle,
		                 markLine=ec.MarkLine()))
		chart.plot()


def  find_activity_cnt(x):
	try:
		x1 = x.split('|')
		return len(x1)
	except:
		return 0

def process_activity_num(x):
    if x>=10:
        return '10+'
    else:
        return int(x)

def count_product_num(filename):
	with open(filename, 'r') as f:
		ecif_product_cnt_dict={}
		num=0
		for line in f:
			line=line.strip('\r\n')
			line=line.split(',')
			if num==0:
				num += 1
				continue
			ecif_product_cnt_dict[int(line[0])]=int(line[1])
	return ecif_product_cnt_dict

def count_product_num_float(filename):
	with open(filename, 'r') as f:
		ecif_product_cnt_dict={}
		num=0
		for line in f:
			line=line.strip('\r\n')
			line=line.split(',')
			if num==0:
				num += 1
				continue
			ecif_product_cnt_dict[int(line[0])]=float(line[1])
	return ecif_product_cnt_dict
def find_pipei_product_cnt(x,ecif_product_cnt_dict):
	try:
		return ecif_product_cnt_dict[int(x)]
	except:return np.nan

def product_num(x):
    if x>=20:
        return '20+'
    else:
        return x

def process_open(x):
	if  x in  ['nan','NULL','null','NAN']:
		return np.nan
	cutoff_now = datetime.datetime.now()
	try:
		x = x.split(" ")[0]
		cutoff_x = datetime.datetime.strptime(x, FORMAT)
		return int(((cutoff_now - cutoff_x).days) / 365)
	except:
		return np.nan

def process_open_time(x):
    if x>=5:
        return 5
    else:
		try:
			return int(x)
		except:
			return x

def product_num_leval(x):
		if x <= 3:
			return '<=3'
		elif x<=6:
			return '>3 & <=6'
		elif x <= 8:
			return '>6 & <=8'
		elif x <= 16:
			return '>8 & <=16'
		elif x <= 20:
			return '>16 & <=20'
		else:
			return '>20'

import datetime
def str_to_time(x):
    time= datetime.datetime.strptime(x, "%Y/%m/%d")
    time_now = datetime.datetime.strptime("2017/6/23", "%Y/%m/%d")
    deltime = (time_now - time).days
    return deltime

def find_loyalty(product_num,open_levals):
	if product_num>8 and open_levals>=2 and open_levals<=3:
		return "buy>8 and 3>=mob>=2"
	else:
		return "other"

def find_loyalty_days(product_num,open_levals,rencent_buy_days,rmb,no_p2p):
	if product_num>8 and open_levals>=2 and rencent_buy_days<365 :
		return "buy>8 and mob>=2 and days<365"
	else:
		return "other"

def load_data():
	##以下均为字典值
	ecif_product_cnt_dict = count_product_num("./data/sf/zh/product_cnt.csv")
	ecif_rencent_days_dict = count_product_num("./data/ecif_id_days.csv")
	ecif_rmb_dict = count_product_num_float("./data/sf/zh/rmb_on_line.csv")
	ecif_product_dict = count_product_num("./data/ecif_id_product.csv")

	##MGM客户
	data = pd.read_csv("./data/sf/zh/final_version_data.csv")
	data['activity_cnt'] = data[['t_activity_name']].apply(lambda x: find_activity_cnt(x[0]), axis=1)
	data['activity_num'] = data[['activity_cnt']].apply(lambda x: process_activity_num(x[0]), axis=1)

	data['product_num'] = data[['ecif_id']].apply(lambda x: find_pipei_product_cnt(x[0],ecif_product_cnt_dict), axis=1)
	data['product_cnt'] = data[['product_num']].apply(lambda x: product_num(x[0]), axis=1)
	data['product_cnt_leval'] = data[['product_cnt']].apply(lambda x: product_num_leval(x[0]), axis=1)

	data['open_levals'] = data[['open_time']].apply(lambda x: process_open(x[0]), axis=1)
	data['open_leval'] = data[['open_levals']].apply(lambda x: process_open_time(x[0]), axis=1)

	data['rencent_days'] = data[['ecif_id']].apply(lambda x: find_pipei_product_cnt(x[0], ecif_rencent_days_dict),axis=1)
	data['rmb'] = data[['ecif_id']].apply(lambda x: find_pipei_product_cnt(x[0], ecif_rmb_dict),axis=1)
	data['no_p2p'] = data[['ecif_id']].apply(lambda x: find_pipei_product_cnt(x[0], ecif_product_dict),axis=1)

	data['loyalty'] = data[['product_num', 'open_levals',]].apply(lambda x: find_loyalty(x[0], x[1]), axis=1)
	data['loyalty_days'] = data[['product_num', 'open_levals', 'rencent_days','rmb','no_p2p']].apply(lambda x: find_loyalty_days(x[0], x[1],x[2],x[3],x[4]),
	                                                                                 axis=1)
	##all 客户
	all_data = pd.read_csv("./data/sf/zh/MGM_feature_nv.csv")

	all_data['activity_cnt'] = all_data[['t_activity_name']].apply(lambda x: find_activity_cnt(x[0]), axis=1)
	all_data['activity_num'] = all_data[['activity_cnt']].apply(lambda x: process_activity_num(x[0]), axis=1)

	all_data['open_levals'] = all_data[['open_time']].apply(lambda x: process_open(x[0]), axis=1)
	all_data['open_leval'] = all_data[['open_levals']].apply(lambda x: process_open_time(x[0]), axis=1)

	all_data['product_num'] = all_data[['ecif_id']].apply(lambda x: find_pipei_product_cnt(x[0], ecif_product_cnt_dict), axis=1)
	all_data['product_cnt'] = all_data[['product_num']].apply(lambda x: product_num(x[0]), axis=1)
	all_data['product_cnt_leval'] = all_data[['product_cnt']].apply(lambda x: product_num_leval(x[0]), axis=1)

	all_data['rencent_days'] = all_data[['ecif_id']].apply(lambda x: find_pipei_product_cnt(x[0], ecif_rencent_days_dict),axis=1)
	all_data['rmb'] = all_data[['ecif_id']].apply(lambda x: find_pipei_product_cnt(x[0], ecif_rmb_dict),axis=1)
	all_data['no_p2p'] = all_data[['ecif_id']].apply(lambda x: find_pipei_product_cnt(x[0], ecif_product_dict),axis=1)

	all_data['loyalty'] = all_data[['product_num','open_levals']].apply(lambda x: find_loyalty(x[0],x[1]), axis=1)
	all_data['loyalty_days'] = all_data[['product_num', 'open_levals', 'rencent_days','rmb','no_p2p']].apply(lambda x: find_loyalty_days(x[0], x[1], x[2],x[3],x[4]),axis=1)

	return data,all_data

def main():
	data,all_data=load_data()
	plot_des(data,all_data)

if __name__ == '__main__':
	main()