#encoding=utf-8
import plot_roc_ks
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import time
from compiler.ast import flatten
import csv

import pandas as pd
import struct
import datetime

class Plot_KS_ROC():
    def __init__(self, groud_truth, y_pred, save_path, name):
        self.groud_truth = groud_truth
        self.y_pred = y_pred
        self.save_path=save_path
        self.name=name

    def plot_roc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.groud_truth, self.y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print  ('auc=%0.5f'%auc)
        fig = plt.figure(figsize=(10, 10))##########
        plt.plot(fpr, tpr, 'r--', linewidth=2.0, aa=False, label='ROC (area=%0.2f)' % (auc))
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
        plt.xlim([0.0, 1.00])
        plt.ylim([0.0, 1.00])
        plt.xlabel('False Postive Rate')
        plt.ylabel('True Postive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        #plt.savefig('%s/roc_curve_%s_v2.png' % (self.save_path,self.name), dpi=180)
        fig.savefig('%s/roc_curve_%s_v2.png' % (self.save_path,self.name), dpi=180)###################
        return auc


    def plot_ks(self):
        groud_truth = np.array(self.groud_truth)
        # index = groud_truth != 1
        # groud_truth[index] = 0  # if not 1, then transform to 0
        order = np.argsort(self.y_pred)  # sort y_pred
        groud_truth_tmp = list(groud_truth[order])  # sort groud_truth according to y_pred

        num_bin = 1000
        len_bin = int(len(groud_truth_tmp) * 1.0 / num_bin)
        #print len(groud_truth_tmp),len_bin
        group = [groud_truth_tmp[i * len_bin:(i + 1) * len_bin] for i in range(0, num_bin)]
        group[-1].extend(groud_truth_tmp[num_bin * len_bin:])
        #print len(group[0]),len(group[1]),len(group[2]),len(group[-1])
        #group[-1] = flatten(group[-1][:])

        total = len(groud_truth_tmp)
        total_good = sum(groud_truth_tmp)
        total_bad = total - total_good
        #print total,total_good


        good_list = [sum(group[i]) for i in range(0, num_bin)]  # number of good for each group
        bad_list = [len(group[i]) - good_list[i] for i in range(0, num_bin)]  # number of bad for each group
        #print  good_list[0:20]
        #print bad_list[900:1000]
        good_ks_result_list = [0]
        bad_ks_result_list = [0]

        for i in range(1, num_bin + 1):
            good_ratio = sum(good_list[0:i]) * 1.0 / total_good
            bad_ratio = sum(bad_list[0:i]) * 1.0 / total_bad
            good_ks_result_list.append(good_ratio)
            bad_ks_result_list.append(bad_ratio)

        diff_list = list(abs(np.array(bad_ks_result_list) - np.array(good_ks_result_list)))
        max_ks_gap_index = diff_list.index(max(diff_list))



        length = len(good_ks_result_list)
        index = range(0, length)
        labels = list(np.array(index) * 1.0 / num_bin)



        fig = plt.figure(figsize=(10, 10))
        axes = fig.gca()
        axes.plot(labels, good_ks_result_list, 'g', linewidth=2, label='good')
        axes.plot(labels, bad_ks_result_list, 'r', linewidth=2, label='bad')
        max_ks_gap_good_value = good_ks_result_list[max_ks_gap_index]

        max_ks_gap_bad_value = bad_ks_result_list[max_ks_gap_index]
        annotate_text_y_index = abs(max_ks_gap_bad_value - max_ks_gap_good_value) / 2 + \
                                min(max_ks_gap_good_value, max_ks_gap_bad_value)
        max_ks_gap_value =max(diff_list)
        xytext_value = str(labels[max_ks_gap_index])
        print ('ks=%s'%max_ks_gap_value)#, max_ks_gap_value, xytext_value

        axes.annotate(xytext_value, xy=(max_ks_gap_index * 1.0 / num_bin, 0),
                      xytext=(max_ks_gap_index * 1.0 / num_bin, 0.05),
                      arrowprops=dict(facecolor='red', shrink=0.05))
        axes.plot([max_ks_gap_index * 1.0 / num_bin, max_ks_gap_index * 1.0 / num_bin],
                  [bad_ks_result_list[max_ks_gap_index], good_ks_result_list[max_ks_gap_index]], linestyle='--',
                  linewidth=2.5)
        axes.annotate(str(round(max_ks_gap_value, 3)), xy=(max_ks_gap_index * 1.0 / num_bin, annotate_text_y_index))
        axes.legend()
        axes.set_title('KS Curve')
        fig.savefig('%s/ks_curve_%s_v2.png' % (self.save_path,self.name), dpi=180)
        plt.close(fig)

def  plot_single_roc_ks(data,every_data,output):


	probability_spit = []

	for i in list(data['probability']):
		# print (i.split('[')[1])
		probability_spit.append(float(1-i))

	plot_obj_test = plot_roc_ks.Plot_KS_ROC(list(data['label']), probability_spit, './plot/',every_data)
	ks_this=plot_obj_test.plot_ks()
	AUC_this=plot_obj_test.plot_roc()
	print "AUC is ",AUC_this   ,"ks is ",ks_this
	output.writerow([every_data,AUC_this,ks_this])

def find_top_N(data,every_data,output):
	percent=[0.05,0.1,0.15]
	length= len(data)
	sort_data = data.sort(columns='new_pro', ascending=False)
	goog_bad=len(sort_data[sort_data['label']==1.0])*1.0/len(sort_data[sort_data['label']==0.0])
	good=len(sort_data[sort_data['label']==1.0])
	pre_result=[every_data,length,good,goog_bad]
	for  val  in percent:
		percent_num= sort_data.head(int(length*val))
		print val,len(percent_num),len(percent_num[percent_num['label']==1.0]),len(percent_num[percent_num['label']==1.0])*1.0/len(percent_num)
		pre_result.extend([len(percent_num),len(percent_num[percent_num['label']==1.0]),len(percent_num[percent_num['label']==1.0])*1.0/len(percent_num)])
	output.writerow(pre_result)

def plot_ks_roc(date_list_test,output,spilt_path):
	for every_data in date_list_test:
		print every_data
		every_df_data=pd.read_csv(spilt_path+every_data, sep='\t', low_memory=False)
		if len(every_df_data)==0:
			print "data length is 0"
		else:
			plot_single_roc_ks(every_df_data,every_data,output)
			# find_top_N(every_df_data,every_data,output)

def output_final(data, Path):
    f = open(Path, "w")
    data.to_csv(Path, sep='\t', index=False, header=True)
    f.close()

def dateRange(start, end, step=1, format="%Y-%m-%d"):
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    days = (strptime(end, format) - strptime(start, format)).days
    return [strftime(strptime(start, format) + datetime.timedelta(i), format) for i in xrange(0, days, step)]

def data_file_name():
	date_list = dateRange('2016-10-02', '2017-06-01')
	return date_list


######## 将全量数据拆分为每天的 ############
def connect_data(date_list, month,path_data, outpath):
    totaldata = pd.read_csv(path_data, sep=',', low_memory=False)
    for i in range(len(date_list)):
        name = str(date_list[i])
        totaldata['new_pro'] = 1 - totaldata['probability']
        subdata = totaldata[totaldata['datadate'] == name]
        output_final(subdata, outpath+name)

if __name__ == '__main__':
	path_data='./data/alading_metrics_out/alading_metrics_v2'
	result_path='./data/alading_metrics_out/result_v2.csv'
	spilt_path='./data/alading_metrics_out/spilt_data_v2/'
	data=pd.read_csv(path_data,sep=',',low_memory=False)

	date_list_test = data_file_name()
	data_file = open(result_path, 'w')

	output = csv.writer(data_file, dialect='excel')
	# output.writerow(['data',"人数",'好人数','好人数目/坏人数目','top_5总人数','top_5好人数','top_5好人占比','top_10总人数','top_10好人数','top_10好人占比','top_15总人数','top_15好人数','top_15好人占比'])
	# pre_result = [every_data, good, length, goog_bad]
	# connect_data(date_list_test, 'month', path_data,spilt_path)

	plot_ks_roc(date_list_test,output,spilt_path)

	# probability_s=[]
	# for i in list(data['probability']):
	# 	# print (i.split('[')[1])
	# 	probability_s.append(float(1 - i))
	#
	# plot_obj_test = plot_roc_ks.Plot_KS_ROC(list(data['label']), probability_s, './plot/', 'all')
	# ks_this = plot_obj_test.plot_ks()
	# AUC_this = plot_obj_test.plot_roc()