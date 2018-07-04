# encoding=utf8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import pandas as pd
import csv
import numpy as np
FORMAT = "%Y-%m-%d"


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
	# print  len(all_data_col[(all_data_col['is_loyalty']=='no_loyalty')&(all_data_col['customer_level']=='3')&(all_data_col['register_province_name']=='other')])

	#划分数据
	data_col0=data_col[data_col['recommend_num'] < 4]

	data_col1 = data_col[data_col['recommend_num'] >=4]

	#得到分母
	length_all=len(all_data_col)
	length_data_col0=len(data_col0)
	length_data_col1 = len(data_col1)

	#3*3交叉集合
	col_list=['sex','open_time','age','is_loyalty','customer_level','customer_source','register_province_name','is_aa','product_cnt']
	print len(col_list)

	#输出到exle
	decrib = ['feature1','val','feature2','val','feature3','val','all_target_percent', '<4_target_percent','>=4_target_percent','<4-all','>=4-all','<4/all','>=4/all',
	          'length_all_target','length_col0_target','length_col1_target', 'length_all','length_data_col0','length_data_col1']
	file0 = open('./data/target2.csv', 'wb+')  # 'wb'
	output = csv.writer(file0, dialect='excel')
	output.writerow(decrib)

	if len(col_list)<3:
		print "col numbers <3"
	else:
		feature_result = get_33_result(col_list)
		print "featuur_3*3 is :",len(feature_result)

		for col_3 in feature_result:
			print col_3
			#三层循环
			for i in list(set(list(data_col0[col_3[0]]))):
				for j in list(set(list(data_col0[col_3[1]]))):
					for k in list(set(list(data_col0[col_3[2]]))):
						#有条件
						length_all_target=target_data_count(all_data_col,col_3[0] ,i,col_3[1],j,col_3[2],k)
						length_col0_target=target_data_count(data_col0, col_3[0], i, col_3[1], j, col_3[2], k)
						length_col1_target=target_data_count(data_col1, col_3[0], i, col_3[1], j, col_3[2], k)
						output.writerow([col_3[0] ,i,col_3[1],j,col_3[2],k,
						                 length_all_target*1.0/length_all, length_col0_target*1.0/length_data_col0, length_col1_target*1.0/length_data_col1,

						                  (length_col0_target * 1.0 / length_data_col0) - (length_all_target * 1.0 / length_all) if length_all_target!=0 else 0 ,
						                 (length_col1_target * 1.0 / length_data_col1)-(length_all_target*1.0/length_all) if length_all_target!=0 else 0,

						                 ((length_col0_target * 1.0 / length_data_col0) / (length_all_target * 1.0 / length_all)) if length_all_target!=0 else 0,
						                 (length_col1_target * 1.0 / length_data_col1) / (length_all_target * 1.0 / length_all) if length_all_target!=0 else 0,

						                  length_all_target,length_col0_target,length_col1_target,
						                 length_all,length_data_col0,length_data_col1])

'''加载相应的数据，提取相应的数据特征'''
def load_data():
    data_fir = "./data/"

    '''step1:取出全量数据和mgm数据，并提取其中相应的数据特征'''
    diff_list=[ 'sex', 'open_time', 'age', 'is_loyalty', 'customer_level', 'customer_source',
     'register_province_name', 'is_aa', 'audit_status', 'card_type', 'current_product']

    mgm_data = pd.read_csv(data_fir + "final_mgm_data_version.csv")
    mgm_data = mgm_data[['ecif_id','recommend_num','sex','open_time','age','is_loyalty','customer_level','customer_source','register_province_name','is_aa','audit_status','card_type','current_product']]
    mgm_data['product_cnt'] = mgm_data['current_product'].astype(np.str).apply(lambda x: len(x.split("|")))

    all_data = pd.read_csv(data_fir + "final_all_data_version.csv")
    all_data = all_data[['ecif_id','recommend_num','sex','open_time','age','is_loyalty','customer_level','customer_source','register_province_name','is_aa','audit_status','card_type','current_product']]
    all_data['product_cnt'] = all_data['current_product'].astype(np.str).apply(lambda x: len(x.split("|")))

    '''step2.2:处理全量数据中数据为空的问题，并填充数据'''
    for column in all_data.columns.tolist():
        all_data.loc[(all_data[column].isnull()), column] = -999

    for column in diff_list:
	    all_data[column] = all_data[column].astype(np.str)
	    mgm_data[column] = mgm_data[column].astype(np.str)

    return mgm_data, all_data

def main():
	data,all_data=load_data()
	# print data,all_data
	plot_des(data,all_data)

if __name__ == '__main__':
	main()