#-*- coding: UTF-8 -*-
import csv
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import numpy as np
def load_data(filename):
    data = pd.read_csv(filename, error_bad_lines=False)
    return data


def find_min_100(train_item):
	item_count = {}
	item_list = list(train_item)
	item_set = list(set(item_list))
	for line in item_set:
		item_count[line] = item_list.count(line)
	sorted(item_count.keys())
	for key,val in item_count.items():
		if val<100:
			return key,val,item_count
	return key,val,item_count

def spilt_bin():
	Spilt_bins={
	'app_cat1_2':4,
	'app_cat1_5':9,
	'app_cat1_6':6,
	'app_cat1_7':11,
	'app_cat1_8':6,
	'app_cat1_10':23,
	'app_cat1_12':5,
	'app_cat1_15':6,
	'app_cat1_16':2,
	'app_cat1_17':4,
	'app_cat1_19':7,
	'app_cat1_20':8,
	'app_cat1_21':9,
	'app_cat1_22':2,
	'app_cat1_24':2,
	'app_cat1_27':2,
	'app_cat1_30':2,
	'app_cat1_31':2,
	'app_cat1_32':7,
	'app_cat1_35':2,
	'app_cat1_36':4,
	'app_cat1_37':13,
	'app_cat1_39':4,
	'app_cat1_41':6,
	'app_cat2_1':2,
	'app_cat2_2':5,
	'app_cat2_5':3,
	'app_cat2_7':3,
	'app_cat2_10':2,
	'app_cat2_12':2,
	'app_cat2_13':3,
	'app_cat2_14':3,
	'app_cat2_15':2,
	'app_cat2_16':3,
	'app_cat2_18':3,
	'app_cat2_19':2,
	'app_cat2_20':4,
	'app_cat2_22':3,
	'app_cat2_24':8,
	'app_cat2_25':2,
	'app_cat2_26':2,
	'app_cat2_27':2,
	'app_cat2_29':2,
	'app_cat2_31':4,
	'app_cat2_32':5,
	'app_cat2_35':2,
	'app_cat2_36':2,
	'app_cat2_38':2,
	'app_cat2_40':3,
	'app_cat2_41':3,
	'app_cat2_43':7,
	'app_cat2_45':7,
	'app_cat2_49':3,
	'app_cat2_50':3,
	'app_cat2_52':3,
	'app_cat2_53':5,
	'app_cat2_54':3,
	'app_cat2_55':2,
	'app_cat2_56':2,
	'app_cat2_59':3,
	'app_cat2_60':2,
	'app_cat2_61':2,
	'app_cat2_68':5,
	'app_cat2_69':2,
	'app_cat2_70':2,
	'app_cat2_71':2,
	'app_cat2_73':5,
	'app_cat2_74':2,
	'app_cat2_77':3,
	'app_cat2_78':2,
	'app_cat2_79':2,
	'app_cat2_82':3,
	'app_cat2_83':2,
	'app_cat2_84':2,
	'app_cat2_87':2,
	'app_cat2_89':4,
	'app_cat2_93':3,
	'app_cat2_94':2,
	'app_cat2_95':3,
	'app_cat2_96':2,
	'app_cat2_97':3,
	'app_cat2_99':3,
	'app_cat2_102':5,
	'app_cat2_108':2,
	'app_cat2_110':3,
	'app_cat2_111':3,
	'app_cat2_112':2,
	'app_cat2_113':3,
	'app_cat2_115':4,
	'app_cat2_118':3,
	'app_cat2_120':6,
	'app_cat2_121':3,
	'app_cat2_124':3,
	'app_cat2_126':2,
	'app_cat2_131':2,
	'app_cat2_132':	5,
	'app_cat2_135':	6,
	'app_cat2_136':	3,
	'app_cat2_138':	2,
	'app_cat2_140':	2,
	'app_cat2_141':	2,
	'app_cat2_142':	3,
	'app_cat2_143':	4,
	'app_cat2_144':	2,
	'app_cat2_148':	2,
	'app_cat2_149':	3,
	'app_cat2_150':	6,
	'app_cat2_152':	2,
	'app_cat2_153':	2,
	'app_cat2_154':	4,
	'app_cat2_156':	2,
	'app_cat2_157':	3,
	'app_cat2_158':	5,
	'app_cat2_159':	3,
	'app_cat2_160':	2,
	'app_cat2_161':	2,
	'app_cat2_162':	2,
	'app_cat2_164':	3,
	'app_cat2_168':	2,
	'app_cat2_170':	2,
	'app_cat2_172':	10,
	'app_cat2_174':	20,
	'app_cat2_175':	3,
	'app_cat2_176':	3,
	'app_cat2_177':	2,
	'app_cat2_178':	3,
	'app_cat2_180':	2,
	'app_cat2_181':	5,
	'app_cat2_184':	2,
	'app_cat2_186':	2,
	'app_cat2_187':	2,
	'app_cat2_190':	2,
	'app_cat2_191':	5,
	'app_cat2_194':	2,
	'app_cat2_195':	4,
	'app_cat2_197':	5,
	'app_cat2_198':	2,
	'app_cat2_202':	3,
	'app_cat2_206':	3,
	'app_cat2_208':	2,
	'app_cat2_211':	3,
	'app_cat2_174': 17,
	'app_cat2_214': 17,
	'app_cat1_45': 30,
	'app_cat1_25': 10
	}
	return Spilt_bins

def Good_ratio_plot(data, col_name, points=[], m=2000):

    data_notnull = data[-data[col_name].isnull()]

    # print data_notnull
    #######################  数据排序  #######################
    index = numpy.argsort(data_notnull[col_name])
    sorted_col=list(pd.Series(list(data_notnull[col_name]))[index])#排序后的列值
    sorted_label=list(pd.Series(list(data_notnull['label']))[index])#排序后的标签
    n=len(sorted_col)

    ###################  排序后, 把值相同的归为一组 ###################
    index2=[0]#the starting location for each group
    i=0
    while i<n-1:
        if(sorted_col[i]==sorted_col[i+1]):
            i=i+1
        else:
            index2.append(i+1)
            i=i+1

    num_group = len(index2)
    #把值相同的归为一组
    group_data=[sorted_col[index2[i]: index2[i + 1]] for i in range(0, num_group - 1)]
    if(index2[-1]==n-1):
        group_data.append([sorted_col[-1]])
    else:
        group_data.append(sorted_col[index2[-1]:n])

    # 按照值的分组把标签分组
    group_label=[sorted_label[index2[i]: index2[i + 1]] for i in range(0, num_group - 1)]
    if(index2[-1]==n-1):
        group_label.append([sorted_label[-1]])
    else:
        group_label.append(sorted_label[index2[-1]:n])

    ##################  计算累计百分比  ###################\
    len_list=[]
    sum_list=[]
    N = []  # 存储累计百分比
    cur_sum = 0
    total_count = len(sorted_col)
    for s in range(0, num_group):
        len_list.append(len(group_data[s]))
        sum_list.append(sum(group_label[s]))
        cur_sum = cur_sum + len(group_data[s])
        N.append(cur_sum * 1.0 / total_count)

    ###################  计算好人比例与密度 ###################
    ratio=[]#存储好人比例
    #density=[]#存储密度
    cur_cnt=len_list[0]
    for first_m in range(0,num_group-1):
        if(cur_cnt<m):
            cur_cnt=cur_cnt+len_list[first_m+1]
        else:
            break

    first_point=first_m/2

    total_cnt1 = len_list[0:first_m+1]
    total_good1 = sum_list[0:first_m+1]
    ratio_1=sum(total_good1)*1.0/sum(total_cnt1)
    for i in range(0,first_point+1):
        ratio.append(ratio_1)
        #density.append(sum(total_cnt1) * 1.0 / (group_data[first_m][0] - group_data[0][0]))

    if(first_m%2==1 and first_m+1<=num_group-1):
        ratio.append((sum(total_good1)+sum_list[first_m+1])*1.0/(sum(total_cnt1)+len_list[first_m+1]))
        #density.append((sum(total_cnt1)+len_list[first_m+1]) * 1.0 / (group_data[first_m+1][0] - group_data[0][0]))
        first_point=first_point+1
        first_m=first_m+1



    forward_pointer = first_m
    backward_pointer = 0

    for i in range(first_point+1,num_group):
        if(len_list[i]>=m):
            ratio.append(sum_list[i]*1.0/len_list[i])
            #density.append(len_list[i] * 1.0 / 0.1)
            forward_pointer=i
            backward_pointer=i
        else:
            if(forward_pointer==num_group -1):
                total_cnt_end = len_list[backward_pointer: forward_pointer + 1]  # 把延伸后的数据合成一个组
                total_good_end = sum_list[backward_pointer: forward_pointer + 1]
                ratio_end = sum(total_good_end) * 1.0 / sum(total_cnt_end)
                for j in range(i, forward_pointer + 1):
                    ratio.append(ratio_end)
                    #density.append(sum(total_cnt_end) * 1.0 / (group_data[forward_pointer][0] - group_data[backward_pointer][0]))
                break
            else:
                if(backward_pointer!=forward_pointer):
                    if (len_list[forward_pointer + 1] > len_list[backward_pointer] or sum(len_list[backward_pointer + 1: forward_pointer + 2]) >= m):
                        forward_pointer = forward_pointer + 1
                        backward_pointer = backward_pointer+1
                    else:
                        forward_pointer = forward_pointer + 2 if forward_pointer + 2 <= num_group - 1 else num_group - 1
                else:
                    forward_pointer = i + 1 if i + 1 <= num_group - 1 else num_group - 1

                total_cnt = len_list[backward_pointer: forward_pointer + 1]  # 把延伸后的数据合成一个组

                total_good = sum_list[backward_pointer: forward_pointer + 1]
                ratio.append(sum(total_good) * 1.0 / sum(total_cnt))  #计算好人比例
                #density.append(sum(total_cnt) * 1.0 / (group_data[forward_pointer][0] - group_data[backward_pointer][0]))


    #画出好人比例图
    fig1 = plt.figure(figsize=(10, 10))
    plt.plot(N, ratio, 'b')
    plt.xlabel(str(col_name))
    plt.ylabel('good ratio')
    #y0 = sum(sorted_label)*1.0/len(sorted_label)
    #plt.axhline(y=y0, linewidth=0.3, color='k')
    for i in range(0,len(points)):
        plt.axvline(x=points[i], linewidth=0.3, color='r')
    # fig1.savefig('fig/'+str(col_name)+'_goodratio' + '.png', dpi=180)

    plt.show()
    print col_name

def mian():
    traindata = load_data('/Users/wanglili/Documents/workstation/working/yrd大赛'
						  '/task1/task1_new_data/task1_data_csv_2/validate.csv')  # [feature_names]

    Spilt_bins=spilt_bin()
    traindata['app_cat1_25'] = np.floor(traindata['app_cat1_25'] / 32)
    traindata['app_cat1_45'] = np.floor(traindata['app_cat1_45'] / 64)
    traindata['app_cat2_174'] = np.floor(traindata['app_cat2_174'] / 16)
    traindata['app_cat2_214'] = np.floor(traindata['app_cat2_214'] / 64)
    traindata = traindata.rename(columns={'tag': 'label'})

    file=open('/Users/wanglili/Documents/workstation/working/yrd大赛'
						  '/task1/task1_new_data/task1_data_csv_2/delete_feature.csv','wb')

    output = csv.writer(file, dialect='excel')
    for item in traindata.columns[2:]:
			print item
			if Spilt_bins.has_key(item):
				print item, "is"
				val=Spilt_bins[item]
				traindata[item][traindata[item]>=val]=val
			else:
				del traindata[item]
				output.writerow([item])



    traindata.to_csv('/Users/wanglili/Documents/workstation/working/yrd大赛'
						  '/task1/task1_new_data/task1_data_csv_2/validate_converted.csv',index=False,header=True)

globals()
if __name__ == '__main__':
	mian()