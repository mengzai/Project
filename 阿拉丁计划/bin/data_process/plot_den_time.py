# encoding=utf8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import pandas as pd
import csv
import numpy as np
FORMAT = "%Y-%m-%d"
import datetime
import echarts as ec


def time_month(x):
	try:
		x = x.split(" ")[0]
		d=x.split("-")[0]+x.split("-")[1]
		return d
	except:
		return x

def count_month(time):
	set_time=sorted(list(set(time)))
	time_cnt=[]
	for mon in set_time:
		time_cnt.append(time.count(mon))
	return set_time,time_cnt


'''加载相应的数据，提取相应的数据特征'''
def load_data():
    data_fir = "./data/"

    '''step1:取出全量数据和mgm数据，并提取其中相应的数据特征'''
    data_name=['activity','order','trade','kyc','communicate']

    for name  in data_name:
	    all_data = pd.read_csv(data_fir +name+ ".csv")
	    all_data = all_data[['event_time']]
	    all_data['time'] = all_data[['event_time']].apply(lambda x: time_month(x[0]), axis=1)
	    set_time, time_cnt=count_month(list(all_data['time']))
	    all_data.drop('activity')
	    # """柱状图实例"""
	    # chart = ec.Echart(True, theme='macarons')
	    # itemStyle = ec.ItemStyle(normal={'label': {'show': 'true', 'position': 'top', 'formatter': '{c}'}})
	    #
	    # chart.use(ec.Title('特征时间分布图', name))
	    # chart.use(ec.Tooltip(trigger='axis'))
	    # # chart.use(ec.Legend(data=['new_data_all', '推荐人数<4', '推荐人数>=4']))
	    # chart.use(ec.Toolbox(show='true', feature=ec.Feature()))
	    # chart.use(ec.Axis(param='x', type='category',
	    #                   data=set_time))
	    # chart.use(ec.Axis(param='y', type='value'))
	    # chart.use(ec.Bar( data=time_cnt, itemStyle=itemStyle,
	    #                  markLine=ec.MarkLine()))
	    # chart.plot()


def main():
	load_data()
	# print data,all_data
	# plot_des(data,all_data)

if __name__ == '__main__':
	main()
