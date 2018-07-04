# encoding=utf8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import pandas as pd
import echarts as ec
FORMAT = "%Y-%m-%d"

def df_to_list(df):
    df = df.tolist()
    res = []
    for each in df:
        vals = [k for k in str(each).split("|") if not pd.isnull(k) and k != 'nan' and k!= 'None' and k!="-999" and k!=""]
        res += list(set(vals))
    return res

def load_data(filename):
	return pd.read_csv(filename, error_bad_lines=False)

def count_product_num(filename):
	with open(filename, 'r') as f:
		ecif_product_all_dict={}
		num=0
		for line in f:
			line=line.strip('\r\n')
			line=line.split(',')
			if num==0:
				num += 1
				continue
			ecif_product_all_dict[int(line[0])]=line[1]
	return ecif_product_all_dict

def find_pipei_product_cnt(x,ecif_product_cnt_dict):
	try:
		return ecif_product_cnt_dict[int(x)]
	except:return np.nan

def value_zhexian1(data,data_set,length):
	data_count = []
	data_list = df_to_list(data)

	for i in data_set:
		data_count.append("%.4f" % (data_list.count(i) * 1.0 / length))
	return data_set,data_count


def value_zhexian(data,length):
	data_count = []
	data_list  = df_to_list(data)
	data_set = list(set(data_list))
	data_set=sorted(data_set)

	for i in data_set:
		data_count.append("%.4f" % (data_list.count(i) * 1.0 / length))
	return data_set,data_count

def  plot_des(data_col,all_data_col):
	data_col0=data_col[data_col['recommend_num'] < 4]
	data_col1 = data_col[data_col['recommend_num'] >=4]
	col_list = ['product_cat']
	for col in col_list:
		data_set, data_count = value_zhexian(all_data_col[col],len(all_data_col))
		data1_set, data1_count = value_zhexian1(data_col0[col], data_set,len(data_col0))
		data2_set, data2_count = value_zhexian1(data_col1[col],data_set, len(data_col1))


		"""柱状图实例"""
		chart = ec.Echart(True, theme='macarons')
		itemStyle = ec.ItemStyle(normal={'label': {'show': 'true', 'position': 'top', 'formatter': '{c}'}})

		chart.use(ec.Title('全量数据与MGM 数据 直方图',col+'current'))
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


def load_all_data():

	all_data = pd.read_csv("./data/sf/zh/MGM_feature_nv.csv")
	data = pd.read_csv("./data/sf/zh/final_version_data.csv")
	ecif_product_set_dict = count_product_num("./data/sf/zh/ecifid_current_product_set.csv")
	all_data['product_cat'] = all_data[['ecif_id']].apply(lambda x: find_pipei_product_cnt(x[0], ecif_product_set_dict), axis=1)
	data['product_cat'] = data[['ecif_id']].apply(lambda x: find_pipei_product_cnt(x[0], ecif_product_set_dict), axis=1)
	return data, all_data


def main():
	data, all_data=load_all_data()
	plot_des(data,all_data)
if __name__ == "__main__":
	main()