# encoding=utf8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import pandas as pd
import numpy as np
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import echarts as ec
import datetime
import csv
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

def target_data_count(target_data,col_numA,i,col_numB,j,col_numC,k):
	return len(target_data[(target_data[col_numA]==i)&(target_data[col_numB]==j)&(target_data[col_numC]==k)])


def  get_33_result(col_list):
	feature_result=[]
	for i in range(len(col_list)):
		for j in range(i+1,len(col_list)):
			for k  in range(j+1,len(col_list)):
				feature_result.append([col_list[i],col_list[j],col_list[k]])
	return feature_result

def  plot_des(data_col,all_data_col):
	data_col0=data_col[data_col['recommend_num'] < 4]
	data_col1 = data_col[data_col['recommend_num'] >=4]
	#得到分母
	length_all=len(all_data_col)
	length_data_col0=len(data_col0)
	length_data_col1 = len(data_col1)

	#3*3交叉集合
	col_list=['sex','open_time','age','is_loyalty','customer_level','customer_source']
	print len(col_list)

	decrib = ['feature1','val','feature2','val','feature3','val','all_target_percent', '<4_target_percent','>=4_target_percent','<4-all','>=4-all','<4/all','>=4/all',
	          'length_all_target','length_col0_target','length_col1_target', 'length_all','length_data_col0','length_data_col1']
	file0 = open('./data/target1.csv', 'wb+')  # 'wb'
	output = csv.writer(file0, dialect='excel')
	output.writerow(decrib)

	if len(col_list)<3:
		print "col numbers <3"
	else:
		feature_result = get_33_result(col_list)
		print "featuur_3*3 is :",len(feature_result)

		for col_3 in feature_result:
			print col_3
			for i in list(set(list(data_col0[col_3[0]]))):
				for j in list(set(list(data_col0[col_3[1]]))):
					for k in list(set(list(data_col0[col_3[2]]))):
						length_all_target=target_data_count(all_data_col,col_3[0] ,i,col_3[1],j,col_3[2],k)
						length_col0_target=target_data_count(data_col0, col_3[0], i, col_3[1], j, col_3[2], k)
						length_col1_target=target_data_count(data_col1, col_3[0], i, col_3[1], j, col_3[2], k)

						print col_3[0] ,i,col_3[1],j,col_3[2],k,length_all_target,length_col0_target,\
							length_col1_target, length_all_target*1.0/length_all, length_col0_target*1.0/length_data_col0,  length_col1_target*1.0/length_data_col1,   length_all,length_data_col0,length_data_col1

						output.writerow([col_3[0] ,i,col_3[1],j,col_3[2],k,
						                 length_all_target*1.0/length_all, length_col0_target*1.0/length_data_col0, length_col1_target*1.0/length_data_col1,

						                  (length_col0_target * 1.0 / length_data_col0) - (length_all_target * 1.0 / length_all) if length_all_target!=0 else 0 ,
						                 (length_col1_target * 1.0 / length_data_col1)-(length_all_target*1.0/length_all) if length_all_target!=0 else 0,

						                 ((length_col0_target * 1.0 / length_data_col0) / (length_all_target * 1.0 / length_all)) if length_all_target!=0 else 0,
						                 (length_col1_target * 1.0 / length_data_col1) / (length_all_target * 1.0 / length_all) if length_all_target!=0 else 0,

						                  length_all_target,length_col0_target,length_col1_target,
						                 length_all,length_data_col0,length_data_col1])

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
	except:return -999

def product_num(x):
    if x>=20:
        return '20+'
    else:
        return x

def process_open(x):
	if  x in  ['nan','NULL','null','NAN']:
		return -999
	cutoff_now = datetime.datetime.now()
	try:
		x = x.split(" ")[0]
		cutoff_x = datetime.datetime.strptime(x, FORMAT)
		return int(((cutoff_now - cutoff_x).days) / 365)
	except:
		return -999

def process_open_time(x):
	if x==-999:
		return x
	elif x>=5:
		return '5+'
	else:
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
		return "loyalty"
	else:
		return "no_loyalty"


'''填充开户时间和生日：注意，这里面有些是缺失数据，所以用-999填充'''
def fill_opentime_birth_date(x):
    if pd.isnull(x):
        return -999
    else:
        return 2017 - int(str(x)[:4])
'''处理省市的问题'''
def process_province(x):
    jiangsu_str = "江苏省"
    zhejiang_str = "浙江省"
    beijing_str = "北京市"
    shanghai_str = "上海市"
    shandong_str = "山东省"
    guangdong_str = "广东省"
    if (x != jiangsu_str) and (x != zhejiang_str) and (x != beijing_str) and (x != shanghai_str) and (
                x != shandong_str):
        return 'other'
    else:
        return x


'''处理客户级别的，将客户分成不同的级别'''
def process_customer_level(x):
    if (x <= 2 and x >= 1):
        return "<=2"
    elif x == 3:
        return "3"
    elif x >= 3:
        return "3+"


'''年龄的划分，这里有可能需要将年龄进一步的合并，特征是30-40岁之间的'''
def process_age(x):
    if x <= 30:
        return "<=30"
    elif x <= 40:
        return "<=40"
    elif x <= 50:
        return "<=50"
    elif x <= 60:
        return "<=60"
    else:
        return ">60"


'''加载相应的数据，提取相应的数据特征'''
def load_data():
    data_fir = "./data/sf/zh/"

    '''step1:取出全量数据和mgm数据，并提取其中相应的数据特征'''
    mgm_data = pd.read_csv(data_fir + "final_version_data.csv")
    mgm_data = mgm_data[
        ["ecif_id", "recommend_num", "sex", "birth_date", "ftrade", "fhighest_education", "customer_level",
         "province_name", "audit_status", "card_type", "customer_source", "fcustomer_status", "finvest_status",
         "open_time","t_activity_name"]]
    all_data = pd.read_csv(data_fir + "MGM_feature_nv.csv")
    # print len(all_data[all_data['fhighest_education'].notnull()])*1.0/len(all_data)
    all_data = all_data[
        ["ecif_id", "recommend_num", "sex", "birth_date", "ftrade", "fhighest_education", "customer_level",
         "province_name", "audit_status", "card_type", "customer_source", "fcustomer_status", "finvest_status",
         "open_time","t_activity_name"]]

    '''step2.1:处理全量数据和mgm数据中的open_time和age问题'''
    mgm_data['age'] = mgm_data['birth_date'].apply(lambda x: 2017 - int(str(x)[:4]))
    mgm_data.drop('birth_date', axis=1, inplace=True)

    all_data['age'] = all_data['birth_date'].apply(lambda x: fill_opentime_birth_date(x))
    all_data.drop('birth_date', axis=1, inplace=True)

    '''step2.2:处理全量数据中数据为空的问题，并填充数据'''
    for column in all_data.columns.tolist():
        all_data.loc[(all_data[column].isnull()), column] = -999

    '''step2.3:处理全量数据和mgm中ftrade以及fhighest_education==-10这部分，其实他们也是空的'''
    mgm_data.loc[(mgm_data['ftrade'] == -10), 'ftrade'] = -999
    mgm_data.loc[(mgm_data['fhighest_education'] == -10), 'fhighest_education'] = -999

    all_data.loc[(all_data['ftrade'] == -10), 'ftrade'] = -999
    all_data.loc[(all_data['fhighest_education'] == -10), 'fhighest_education'] = -999

    # print len(all_data[all_data['fhighest_education']==-999])*1.0/len(all_data)

    '''step2.4:处理全量数据和mgm数据中的customer_level和open_time,province_name以及age离散化的问题'''
    mgm_data['customer_level'] = mgm_data['customer_level'].apply(lambda x: process_customer_level(x))
    mgm_data['province_name'] = mgm_data['province_name'].apply(lambda x: process_province(x))
    mgm_data['age'] = mgm_data['age'].apply(lambda x: process_age(x))

    all_data['customer_level'] = all_data['customer_level'].apply(lambda x: process_customer_level(x))
    all_data['province_name'] = all_data['province_name'].apply(lambda x: process_province(x))
    all_data['age'] = all_data['age'].apply(lambda x: process_age(x))

    ##以下均为字典值
    ecif_product_cnt_dict = count_product_num("./data/sf/zh/product_cnt.csv")
    ecif_rencent_days_dict = count_product_num("./data/ecif_id_days.csv")
    ecif_rmb_dict = count_product_num_float("./data/sf/zh/rmb_on_line.csv")
    ecif_product_dict = count_product_num("./data/ecif_id_product.csv")

    mgm_data['activity_cnt'] = mgm_data[['t_activity_name']].apply(lambda x: find_activity_cnt(x[0]), axis=1)
    mgm_data['activity_num'] = mgm_data[['activity_cnt']].apply(lambda x: process_activity_num(x[0]), axis=1)

    mgm_data['product_num'] = mgm_data[['ecif_id']].apply(lambda x: find_pipei_product_cnt(x[0], ecif_product_cnt_dict), axis=1)
    mgm_data['product_cnt'] = mgm_data[['product_num']].apply(lambda x: product_num(x[0]), axis=1)
    mgm_data['product_cnt_leval'] = mgm_data[['product_cnt']].apply(lambda x: product_num_leval(x[0]), axis=1)

    mgm_data['open_levals'] = mgm_data[['open_time']].apply(lambda x: process_open(x[0]), axis=1)
    mgm_data['open_time'] = mgm_data[['open_levals']].apply(lambda x: process_open_time(x[0]), axis=1)

    mgm_data['rencent_days'] = mgm_data[['ecif_id']].apply(lambda x: find_pipei_product_cnt(x[0], ecif_rencent_days_dict),
                                                   axis=1)
    mgm_data['rmb'] = mgm_data[['ecif_id']].apply(lambda x: find_pipei_product_cnt(x[0], ecif_rmb_dict), axis=1)
    mgm_data['no_p2p'] = mgm_data[['ecif_id']].apply(lambda x: find_pipei_product_cnt(x[0], ecif_product_dict), axis=1)

    mgm_data['loyalty'] = mgm_data[['product_num', 'open_levals', ]].apply(lambda x: find_loyalty(x[0], x[1]), axis=1)
    mgm_data['is_loyalty'] = mgm_data[['product_num', 'open_levals', 'rencent_days', 'rmb', 'no_p2p']].apply(
	    lambda x: find_loyalty_days(x[0], x[1], x[2], x[3], x[4]),
	    axis=1)


    all_data['activity_cnt'] = all_data[['t_activity_name']].apply(lambda x: find_activity_cnt(x[0]), axis=1)
    all_data['activity_num'] = all_data[['activity_cnt']].apply(lambda x: process_activity_num(x[0]), axis=1)

    all_data['open_levals'] = all_data[['open_time']].apply(lambda x: process_open(x[0]), axis=1)
    all_data['open_time'] = all_data[['open_levals']].apply(lambda x: process_open_time(x[0]), axis=1)

    all_data['product_num'] = all_data[['ecif_id']].apply(lambda x: find_pipei_product_cnt(x[0], ecif_product_cnt_dict),axis=1)
    all_data['product_cnt'] = all_data[['product_num']].apply(lambda x: product_num(x[0]), axis=1)
    all_data['product_cnt_leval'] = all_data[['product_cnt']].apply(lambda x: product_num_leval(x[0]), axis=1)

    all_data['rencent_days'] = all_data[['ecif_id']].apply(lambda x: find_pipei_product_cnt(x[0], ecif_rencent_days_dict), axis=1)
    all_data['rmb'] = all_data[['ecif_id']].apply(lambda x: find_pipei_product_cnt(x[0], ecif_rmb_dict), axis=1)
    all_data['no_p2p'] = all_data[['ecif_id']].apply(lambda x: find_pipei_product_cnt(x[0], ecif_product_dict), axis=1)

    # all_data['loyalty'] = all_data[['product_num', 'open_levals']].apply(lambda x: find_loyalty(x[0], x[1]), axis=1)
    all_data['is_loyalty'] = all_data[['product_num', 'open_levals', 'rencent_days', 'rmb', 'no_p2p']].apply(lambda x: find_loyalty_days(x[0], x[1], x[2], x[3], x[4]), axis=1)

    # all_data[['ecif_id','is_loyalty']].to_csv('./data/all_loyalty.csv')
    # mgm_data[['ecif_id', 'is_loyalty']].to_csv('./data/mgm_loyalty.csv')

    return mgm_data, all_data
def main():
	data,all_data=load_data()
	plot_des(data,all_data)

if __name__ == '__main__':
	main()