# encoding=utf8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import echarts as ec
import datetime
FORMAT = "%Y-%m-%d"
from pandas.core.frame import DataFrame

def plot_dispersed():
	name_important=pd.read_csv('name_important.csv')

	sort_fea=name_important.sort(['important'],ascending=False)
	# print(sort_fea)
	print(sort_fea.head(20)[["name","important"]])

	name=list(sort_fea.head(20)["name"])
	important=list(sort_fea.head(20)["important"])

	impo=[]
	for i in important:
		impo.append('%.4f' % i)

	chart = ec.Echart(True, theme='macarons')
	chart.use(ec.Title('Feature_important', x='center'))
	chart.use(ec.Tooltip(trigger='axis'))
	chart.use(ec.Toolbox(show='true', feature=ec.Feature()))
	chart.use(ec.Axis(param='x', type='category', data=name, axisLabel={'interval': 0, 'rotate': -20}))
	chart.use(ec.Axis(param='y', type='value'))
	itemStyle = ec.ItemStyle(
		normal={
			'label': {
				'show': 'true',
				'position': 'top',
				'formatter': '{c}'}})
	chart.use(
		ec.Bar(
			data=impo,
			itemStyle=itemStyle))
	chart.plot()

if __name__ == '__main__':
	plot_dispersed()

