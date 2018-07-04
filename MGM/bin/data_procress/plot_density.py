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


parser = argparse.ArgumentParser()
parser.add_argument('--data_name',type=str,default='data/sf/zh/MGM_feature_v5_my.csv',help='training data in csv format')
args = parser.parse_args()

def outliers_detection(data, times = 7, quantile = 0.95):
    data=data[-data.isnull()]
    data = np.array(sorted(data))
    #std-outlier
    outlier1 = np.mean(data) + 1*np.std(data)

    # mad-outlier
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    outlier2 = med + times * mad

    # quantile-outlier
    outlier3 = data[int(np.floor(quantile * len(data)) - 1)]
    return outlier1, outlier2, outlier3

def density(new_data_all,new_data_MGM,col_name,fig_save_path):
    new_data_all1 = new_data_all[-new_data_all[col_name].isnull()][col_name]
    new_data_MGM1 = new_data_MGM[-new_data_MGM[col_name].isnull()][col_name]

	#加入范围
    # data_not_outliers1=data_notnull1[data_notnull1<=outliers_max]
    sns.distplot(new_data_all1, rug=False, hist=False, label='all_')
    sns.distplot(new_data_MGM1, rug=False, hist=False, label='MGM_')
    plt.savefig(str(fig_save_path)  + "_" + str(col_name) + '.png', dpi=180)
    plt.close()

def load_data(path):
    return pd.read_csv(path, error_bad_lines=False)

def plot_density(new_data_all,new_data_MGM):
    colom_name_list=['age_leval','open_leval','recommend_num']
    print len(colom_name_list)
    for m in range(0, len(colom_name_list)):
        colom = colom_name_list[m]
        print  colom,new_data_all[colom]
        # print " colom   name  is :",colom_name_list[m]

        #choose the fig_save_path
        SP = "density"
        fig_save_path='compar/all_ori/'
        fig_save_path = str(fig_save_path) + str(colom)

        # try:

            # # choose best outlier
            # outlier1, outlier2, outlier3 = outliers_detection(new_data[colom])
            # # print outlier2
            # outlier2=np.mean([outlier1, outlier2, outlier3])

        density(new_data_all,new_data_MGM, colom, fig_save_path)
            # break
        # except:
        #     print "err",colom


def value_zhexian1(data,data_set):
	data_count = []
	data_list = list(data)
	length=len(data_list)

	for i in data_set:
		data_count.append("%.4f" % (data_list.count(i)*1.0/length))
	return data_set,data_count


def plot_dispersed(new_data_all,new_data_MGM):
    colom_name_list=['sex','ftrade','fhighest_education','customer_level',
					 'audit_status','card_type','customer_source','fcustomer_status','finvest_status']
    print len(colom_name_list)
    for m in range(0, len(colom_name_list)):
        col_name = colom_name_list[m]

        new_data_all.loc[(new_data_all[col_name].isnull()), col_name] = np.nan
        new_data_MGM.loc[(new_data_MGM[col_name].isnull()), col_name] = np.nan

        new_data_all.loc[(new_data_all[col_name]==-10), col_name] = np.nan
        new_data_MGM.loc[(new_data_MGM[col_name]==-10), col_name] = np.nan


        new_data_all1 = new_data_all[-new_data_all[col_name].isnull()][col_name]
        new_data_MGM1= new_data_MGM[-new_data_MGM[col_name].isnull()][col_name]

        data_set, data_count = value_zhexian_set_count(new_data_all1)
        print data_set,data_count,col_name
        data1_set, data1_count = value_zhexian1(new_data_MGM1, data_set)
        print data1_set, data1_count
        """柱状图实例"""
        chart = ec.Echart(True, theme='macarons')
        chart.use(ec.Title('全量数据与MGM 数据 直方图', col_name))
        chart.use(ec.Tooltip(trigger='axis'))
        chart.use(ec.Legend(data=['new_data_all', 'new_data_MGM']))
        chart.use(ec.Toolbox(show='true', feature=ec.Feature()))
        chart.use(ec.Axis(param='x', type='category',
			   data=data_set))
        chart.use(ec.Axis(param='y', type='value'))
        chart.use(ec.Bar(name='new_data_all', data=data_count,
			  markPoint=ec.MarkPoint(), markLine=ec.MarkLine()))
        chart.use(ec.Bar(name='new_data_MGM', data=data1_count,
			  markLine=ec.MarkLine()))
        chart.plot()

def value_zhexian(data):
	data_count = []
	data_list = list(data)
	length=len(data_list)
	data_set = list(set(data_list))
	data_set.sort()

	count={}
	for i in data_set:
		count[i]=data_list.count(i)
		data_count.append(data_list.count(i))
	l = sorted(count.items(), key=lambda d: d[1],reverse=True)
	# l= l[:10]

	return l,count
	#
	# chinese_name=[]
	#
	# final_set=[]
	# final_count=[]
	# for key,val  in l:
	# 	final_set.append(key)
	# 	final_count.append(val)
	# 	chinese_name.append(name[str(int(key))])
	#
	# return final_set,final_count,chinese_name
def value_zhexian_set_count(data):
	data_count = []
	data_list = list(data)
	length=len(data_list)
	data_set = list(set(data_list))
	data_set=sorted(data_set)

	for i in data_set:
		data_count.append(data_list.count(i)*1.0/length)
	return data_set,data_count

def plot_final(train,item,path,name):

	newdata_train = train[-train.isnull()]
	fig = plt.figure(figsize=(20, 10))
	train_set, train_count,chinese_name=value_zhexian(newdata_train,name)

	plt.plot(train_set, train_count,label=item)
	plt.title(item)
	plt.legend()
	plt.show()

def zhexian(name):
	data_name = args.data_name
	new_data = load_data(data_name)
	fea_list = new_data.columns

	for m in range(0, len(colom_name_list)):
		colom = 'code'
		print colom
		fig_save_path = 'fig_zhexian/'
		fig_save_path = str(fig_save_path)
		plot_final(new_data[colom],colom,fig_save_path,name)

		break

def load_target_file(file):
	count = 0
	colom_name_list=[]
	colom_name_list.extend(['code','name','cus_age','cus_open_time'])


	with open(file, 'r') as f:
		for line in f:
			line = line.split(',')
			if count==0:
				count += 1
				continue
			if float(line[1])>=0.50:
				colom_name_list.append(line[0])
			count += 1

	return colom_name_list

def value_aum(data):
	data_count = []
	data_list = list(np.log10(data))
	data_set = list(set(data_list))

	for i in data_set:
		data_count.append(data_list.count(i))
	return data_set,data_count

def eop_aum(new_data_MGM,new_data_all):
	data_name = args.data_name
	new_data = load_data(data_name)
	fea_list = new_data.columns

	for m in ['aum','eop','favailable_funds']:
		colom = m

		fig_save_path = 'compar/all_ori/'
		try:
			train=new_data_MGM[colom]
			newdata_train = train[-train.isnull()]
			newdata_train = newdata_train[newdata_train > 1000000]
			# newdata_train = newdata_train[newdata_train >= 100]

			train1 = new_data_all[colom]
			newdata_train1 = train1[-train1.isnull()]
			newdata_train1 = newdata_train1[newdata_train1 > 1000000]
			# newdata_train1 = newdata_train1[newdata_train1 >= 100]

			# newdata_train = newdata_train[newdata_train >1000000]
			newdata_train=np.log10(newdata_train)
			newdata_train1 = np.log10(newdata_train1)

			sns.distplot(newdata_train, rug=False, hist=False, label='mgm_'+colom)
			sns.distplot(newdata_train1, rug=False, hist=False, label='all_'+colom)

			plt.savefig(str(fig_save_path) + str(colom) + '_>100万_' + '.png', dpi=180)
			plt.close()
		except:
			print len( newdata_train[newdata_train > 1000000])
			print "err",colom

def  load_target_dict(file):
	with open(file, 'r') as f:
		name={}
		for line in f:
			line = line.split(',')
			name[line[0]]=line[1]
	return name

def plot_bar(new_data):
	name = load_target_dict(data_name)
	newdata_train = new_data['province']
	l = value_zhexian(newdata_train, name)
	labels = []
	ind = []
	TSA = []
	cnt = 0
	for i in l:
		ind.append(cnt)
		cnt += 1
		TSA.append(i[1])
		print name[str(int(i[0]))]
		labels.append(name[str(int(i[0]))])

	fig = plt.figure(1, figsize=(10, 8))
	ax = fig.add_subplot(111)
	ax.bar(ind, TSA, 0.3, color='b', label='bar')
	ax.set_xticks(ind)
	ax.set_xticklabels(labels)
	plt.show()
	fig.savefig('./data/sf/zh/sheng' + '.png')

def plot_pencent_bar(new_data_MGM,new_data_all):
	name = load_target_dict('./data/sf/zh/MGM_feature_nv_new.csv')
	# print name
	new_data_MGM_province = new_data_MGM['province']
	new_data_all_province = new_data_all['province']

	print len(new_data_MGM_province),len(new_data_all_province)
	MGM_l,count_MGM = value_zhexian(new_data_MGM_province)
	all_l,count_all = value_zhexian(new_data_all_province)

	print MGM_l,count_MGM,len(MGM_l)
	print all_l,count_all,len(all_l)

	# print MGM_l[:10],all_l[:10]

	final_pro=[]
	for  i  in MGM_l[:10]:
		final_pro.append(i[0])
	for j  in all_l[:10]:
		final_pro.append(j[0])

	set_final_pro= list(set(list(final_pro)))

	# print name
	# for va  in count_all:
	# 	print name[str(va)]

	all_pencent={}
	mgm_pencent = {}
	for val  in set_final_pro:
		all_pencent[val]=[count_all[val]*1.0,count_MGM[val]*1.0]
	l = sorted(all_pencent.items(), key=lambda d: d[1], reverse=True)

	labels = []
	ind = []
	TSA1 = []
	TSA2 = []
	cnt = 0
	for i in l:
		print i
		ind.append(cnt)
		cnt += 1
		TSA1.append(i[1][0])
		TSA2.append(i[1][1])
		print name[str(int(i[0]))]
		labels.append(name[str(int(i[0]))])

	print ind,TSA1,TSA2,labels

	"""柱状图实例"""
	chart = ec.Echart(True, theme='macarons')
	chart.use(ec.Title('全量数据与MGM 数据 直方图','省'))
	chart.use(ec.Tooltip(trigger='axis'))
	chart.use(ec.Legend(data=['new_data_all', 'new_data_MGM']))
	chart.use(ec.Toolbox(show='true', feature=ec.Feature()))
	chart.use(ec.Axis(param='x', type='category',
					  data=labels))
	chart.use(ec.Axis(param='y', type='value'))
	chart.use(ec.Bar(name='new_data_all', data=TSA1,
					 markPoint=ec.MarkPoint(), markLine=ec.MarkLine()))
	chart.use(ec.Bar(name='new_data_MGM', data=TSA2,
					 markLine=ec.MarkLine()))
	chart.plot()

	# fig = plt.figure(1, figsize=(10, 8))
	# ax = fig.add_subplot(111)
	# ax.bar(ind, TSA, 0.3, color='b', label='bar')
	# ax.set_xticks(ind)
	# ax.set_xticklabels(labels)
	# plt.show()
	# fig.savefig('./data/sf/zh/sheng' + '.png')


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



def  plot_t_recommend_num(new_data_MGM,new_data_all):
	colom_name_list = ['sex', 'age_leval', 'ftrade', 'fhighest_education',
					   'customer_level','']
	print len(colom_name_list)

	for m in range(0, len(colom_name_list)):
		col_name = colom_name_list[m]
		print col_name

		new_data_MGM.loc[(new_data_MGM[col_name].isnull()), col_name] = np.nan

		new_data_MGM.loc[(new_data_MGM[col_name] == -10), col_name] = np.nan

		new_data_MGM1 = new_data_MGM[-new_data_MGM[col_name].isnull()]

		new_data_all.loc[(new_data_all[col_name].isnull()), col_name] = np.nan

		new_data_all.loc[(new_data_all[col_name] == -10), col_name] = np.nan

		new_data_all1 = new_data_all[-new_data_all[col_name].isnull()]


		data1_set, data1_count = value_zhexian_set_count(new_data_all1[col_name])
		print data1_set, data1_count

		# data1, recommend_num1 = value_zhexian1(new_data_MGM1[new_data_MGM1['recommend_num']==1][col_name], data1_set)
		# print data1, recommend_num1
		#
		# data2, recommend_num2 = value_zhexian1(new_data_MGM1[new_data_MGM1['recommend_num'] == 2][col_name], data1_set)
		# print data2, recommend_num2
		#
		# data3, recommend_num3 = value_zhexian1(new_data_MGM1[new_data_MGM1['recommend_num'] == 3][col_name], data1_set)
		# print data3, recommend_num3
		#
		# data4, recommend_num4 = value_zhexian1(new_data_MGM1[new_data_MGM1['recommend_num'] == 4][col_name], data1_set)
		# print data4, recommend_num4
		#
		# data5, recommend_num5 = value_zhexian1(new_data_MGM1[new_data_MGM1['recommend_num'] >4][col_name], data1_set)
		# print data5, recommend_num5
		#
		# data6, recommend_num6 = value_zhexian1(new_data_all1[col_name], data1_set)
		# print data6, recommend_num6

		# data0, recommend_num0 = value_zhexian_set_count(new_data_all1[col_name])
		# print data0, recommend_num0

		new_data_MGM2=new_data_MGM1[new_data_MGM1['open_leval']<2]
		data7, recommend_num7 = value_zhexian1(new_data_MGM2[new_data_MGM2['recommend_num'] >= 4][col_name], data1_set)
		print len(new_data_MGM2[new_data_MGM2['recommend_num'] >= 4][col_name]),data7, recommend_num7
		new_data_all2 = new_data_all1[new_data_all1['open_leval'] < 2]
		data8, recommend_num8= value_zhexian1(new_data_all2[col_name], data1_set)
		print len(new_data_all2[col_name]),data8, recommend_num8


		"""柱状图实例"""
		chart = ec.Echart(True, theme='macarons')
		chart.use(ec.Title('mgm 推荐人数差异', col_name))
		chart.use(ec.Tooltip(trigger='axis'))
		chart.use(ec.Legend(data=['mgm  mob<2 推荐人数>=4', 'all  mob<2 推荐人数>=4']))
		chart.use(ec.Toolbox(show='true', feature=ec.Feature()))
		chart.use(ec.Axis(param='x', type='category',
						  data=data1_set))
		chart.use(ec.Axis(param='y', type='value'))
		# chart.use(ec.Bar(name='推荐人数=1', data=recommend_num1,
		# 				 markPoint=ec.MarkPoint(), markLine=ec.MarkLine()))
		# chart.use(ec.Bar(name='推荐人数=2', data=recommend_num2,
		# 				 markLine=ec.MarkLine()))
		# chart.use(ec.Bar(name='推荐人数=3', data=recommend_num3,
		# 				 markLine=ec.MarkLine()))
		# chart.use(ec.Bar(name='推荐人数=4', data=recommend_num4,
		# 				 markLine=ec.MarkLine()))
		# chart.use(ec.Bar(name='推荐人数>4', data=recommend_num5,
		# 				 markLine=ec.MarkLine()))
		#
		# chart.use(ec.Bar(name='全部数据', data=recommend_num6,
		# 				 markLine=ec.MarkLine()))

		chart.use(ec.Bar(name='mgm  mob<2 推荐人数>=4', data=recommend_num7,
						 markLine=ec.MarkLine()))

		chart.use(ec.Bar(name='all  mob<2 推荐人数>=4', data=recommend_num8,
						 markLine=ec.MarkLine()))
		chart.plot()


def process_age_spilt(x):
    if x<=30:
        return "<=30"
    elif x<=40:
        return "<=40"
    elif x<=50:
        return "<=50"
    elif x<=60:
        return "<=60"
    else:
        return ">60"

def process_customer_level(x):
    if (x<=2 and x>=1) :
        return "<=2"
    elif x==3:
        return "3"
    elif x>=3:
        return "3+"


jiangsu_str="江苏省"
zhejiang_str="浙江省"
beijing_str="北京市"
shanghai_str="上海市"
shandong_str="山东省"
guangdong_str="广东省"

def process_open_time(x):
    if x>=5:
        return 5
    else:
		try:
			return int(x)
		except:
			return x

def process_province(x):
    if (x!=jiangsu_str) and (x!=zhejiang_str) and (x!=beijing_str) and (x!=shanghai_str) and (x!=shandong_str):
        return 'other'
    else:
        return x


def process_age(x):
    if x in ['nan', 'NULL', 'null', 'NAN']:
        print x =='nan'
        return np.nan
    cutoff_now=datetime.datetime.now()

    try:
        cutoff_x = datetime.datetime.strptime(x,FORMAT)
        return int (((cutoff_now-cutoff_x).days)/365)
    except:
		return np.nan

def  describe(new_data_MGM,new_data_all):
	fea_list = new_data_all.columns
	colom_name_list = [
		'ecif_id',
		'customer_code',
		'frecommend_id',
		'recommend_num',
		't_frecommend_depid',
		't_frecommend_cityid',
		't_frecommend_yinyebuid',
		'aum',
		'eop',
		'customer_name',
		'mobile',
		'tel',
		'mail',
		'card_type',
		'card_no',
		'sex',
		'birth_date',
		'province',
		'city',
		'customer_level',
		'status',
		'open_status',
		'audit_status',
		'invest_status',
		'crm',
		'cca',
		'customer_source',
		'customer_source2',
		'customer_source3',
		'open_time',
		't_customer_hobby_code',
		't_customer_hobby_desc',
		'fmarital_state',
		'fintention',
		'fclient_language',
		'foffice_phone',
		'ffirst_contact_date',
		'fclient_contact_time',
		'fstage',
		'favailable_funds',
		'ffinancial_state',
		'fcard_issue_organ',
		'fprefecture',
		'ftrade',
		'fwork_unit',
		'fposition',
		'fclient_hobby',
		'finvest_status',
		'fcustomer_status',
		'frank',
		'fhighest_education',
		'fpeople',
		'fchildren_count',
		'fnationality',
		'fstreet',
		'fcensus_register_country',
		'fcensus_register_province',
		'fcensus_register_city',
		'fcensus_register_prefecture',
		'fcensus_register_street',
		't_activity_name',
		't_activity_no',
		't_signup_state',
		't_activity_target',
		't_activity_mode',
		't_activity_address',
		't_plan_start_time',
		't_plan_end_time',
		't_real_start_time',
		't_real_end_time',
		't_activity_scale',
		't_plan_costs',
		't_real_costs',
		't_product_type_tag',
		't_customer_level_tag'
	]
	print len(colom_name_list)
	for m in range(0, len(colom_name_list)):
		colom = colom_name_list[m]

		data_notnull = new_data_MGM[-new_data_MGM[colom].isnull()][colom]
		data_notnull1 = new_data_all[-new_data_all[colom].isnull()][colom]
		print len(data_notnull)*1.0/len(new_data_MGM)

	# colom_name_list = ['open_leval', 'finvest_status', 'fcustomer_status', 'customer_source','fhighest_education',
	# 				   'card_type','province_name','audit_status','card_type','customer_source','fcustomer_status',
	# 				   'finvest_status']
	# print len(colom_name_list)
	#
	# for m in range(0, len(colom_name_list)):
	# 	colmon=colom_name_list[m]
	# 	new_data_MGM1 = new_data_MGM[-new_data_MGM[colmon].isnull()][colmon]
	# 	print colmon,len(new_data_MGM1)*1.0/len(new_data_MGM)
if __name__ == '__main__':
	data_name = args.data_name
	colom_name_list=load_target_file('data/fugailv.csv')
	new_data_MGM = load_data('data/sf/zh/MGM_feature_v5_my.csv')

	new_data_all = load_data('./data/sf/zh/MGM_feature_nv.csv')

	new_data_MGM['age_leval'] = new_data_MGM[['birth_date']].apply(lambda x: process_age_spilt(x[0]), axis=1)
	new_data_MGM['open_levals'] = new_data_MGM[['open_time']].apply(lambda x: process_open(x[0]), axis=1)
	new_data_MGM['open_leval'] = new_data_MGM[['open_levals']].apply(lambda x: process_open_time(x[0]), axis=1)

	new_data_all['age_leval'] = new_data_all[['birth_date']].apply(lambda x: process_age_spilt(x[0]), axis=1)
	new_data_all['open_levals'] = new_data_all[['open_time']].apply(lambda x: process_open(x[0]), axis=1)
	new_data_all['open_leval'] = new_data_all[['open_levals']].apply(lambda x: process_open_time(x[0]), axis=1)

	# new_data_all['age_leval'] = new_data_all[['birth_date']].apply(lambda x: process_age(x[0]), axis=1)
	# new_data_all['open_leval'] = new_data_all[['open_time']].apply(lambda x: process_open(x[0]), axis=1)

	# print len(new_data_all)
	# eop_aum(new_data_MGM,new_data_all)
	#plot bar
	# plot_pencent_bar(new_data_MGM,new_data_all)
	plot_dispersed(new_data_all,new_data_MGM)
	# plot_density(new_data_all,new_data_MGM)

	describe(new_data_MGM,new_data_all)

	plot_t_recommend_num(new_data_MGM,new_data_all)







