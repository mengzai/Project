# encoding=utf8

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import echarts as ec
FORMAT = "%Y-%m-%d"
from pandas.core.frame import DataFrame

def plot_dispersed():
	name=list([
		"register_province",
		"live_province"	,
		"marriage"   ,
		"credit_high_limit"  ,
		"org_province"   ,
		"live_city" ,
		"gender"   ,
		"register_city" ,
		"identity_address_province"	 ,
		"credit_grade"
	])
	important=list(
[
	0.1278,
	0.1249,
	0.1198,
	0.0569,
	0.0561,
	0.0544,
	0.0445,
	0.0443,
	0.0352,
	0.0326
]
)
	"""柱状图实例"""
	chart = ec.Echart(True, theme='macarons')
	chart.use(ec.Toolbox(show='true', feature=ec.Feature()))

	chart.use(ec.Title('特征属性描述' ))
	chart.use(ec.Tooltip(trigger='axis'))

	# chart.use(ec.Legend(data=['沟通']))
	chart.use(ec.Toolbox(show='true', feature=ec.Feature()))

	wargs = {"axisLabel": {"interval": 0, "rotate": 5}}
	chart.use(ec.Axis(param='x', type='category',
	                  data=name, **wargs
	                  ))

	chart.use(ec.Axis(param='y', type='value'))
	itemStyle = ec.ItemStyle(
		normal={
			'label': {
				'show': 'true',
				'position': 'top',
				'formatter': '{c}'}})
	chart.use(ec.Bar(name='new_data_all', data=important, itemStyle=itemStyle,
			barWidth=25,
			markPoint=None,
			markLine=None))

	chart.plot()
if __name__ == '__main__':
	plot_dispersed()

