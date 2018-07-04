# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
reload(sys)
sys.setdefaultencoding('utf-8') 
import datetime
FORMAT = "%Y-%m-%d"
from matplotlib import pyplot as plt
from collections import Counter

def df_to_list(df):
    df = df.tolist()
    res = []
    for each in df:
        vals = [k for k in str(each).split("|") if not pd.isnull(k) and k != 'nan' and k!= 'None' and k!="-999" and k!=""]
        res += list(set(vals))
    return res

def barplot(data=[], savename='bar.jpg', most_common=10, cnt=1):
    cntdict = Counter(data)
    cnttuple = map(lambda x: (x[0], x[1]*1.0/cnt), cntdict.most_common(most_common))
    index = range(len(cnttuple))
    values = [val[1] for val in cnttuple]
    xticks = [val[0] for val in cnttuple]
    plt.bar(index, values)
    plt.title(savename.split('.')[0])
    plt.xticks(index, xticks)
    if savename:
        plt.savefig(savename)
    # plt.show()
    plt.close()

def pieplot(data=[], savename='pie.jpg', most_common=10, cnt=1):
    cntdict = Counter(data)
    cnttuple = map(lambda x: (x[0], x[1] * 1.0 / cnt), cntdict.most_common(most_common))
    index = range(len(cnttuple))
    values = [val[1] for val in cnttuple]
    xticks = [val[0] for val in cnttuple]
    plt.pie(values, labels=xticks, autopct='%1.2f%%')
    plt.title(savename.split('.')[0])
    # plt.xticks(index, xticks)
    if savename: 
        plt.savefig(savename)
    #plt.show()
    plt.close()

def load_data(filename):
	return pd.read_csv(filename, error_bad_lines=False)

def process_age(x):
    cutoff_now=datetime.datetime.now()

    cutoff_x = datetime.datetime.strptime(x,FORMAT)
    return int (((cutoff_now-cutoff_x).days)/365)

def df_spit(final_data):
	sex_1_age_leval30 = final_data[(final_data["sex"] ==1) &  (final_data['age_leval'] <= 60)&  (final_data['age_leval'] > 30)]
	return sex_1_age_leval30['fn_hobbies']

def fugailv():
	data = load_data("./data/sf/zh/new_data.csv")
	data['age_leval'] = data[['birth_date']].apply(lambda x: process_age(x[0]), axis=1)

	data.loc[(data['fn_hobbies'] == '-999'), 'fn_hobbies'] = np.nan
	data.loc[(data['app_hobby'] == '-999'), 'app_hobby'] = np.nan
	data.loc[(data['kyc_hobby'] == '-999'), 'kyc_hobby'] = np.nan
	data.loc[(data['activity_hobby'] == '-999'), 'activity_hobby'] = np.nan
	data.loc[(data['yl_hobby'] == '-999'), 'yl_hobby'] = np.nan
	data.loc[(data['plane_hobby'] == '-999'), 'plane_hobby'] = np.nan
	hobby_list = ["kyc_hobby", "app_hobby", "activity_hobby", "yl_hobby", "plane_hobby", "fn_hobbies"]

	fugai=[]
	data_1 = data[data["sex"] == 0]
	data_1 = data_1[data_1["age_leval"] <=30]
	fugai.append(data_1)

	data_2 = data[data["sex"] == 0]
	data_2 = data_2[data_2["age_leval"] > 60]
	fugai.append(data_2)

	data_3 = data[data["sex"] == 0]
	data_3 = data_3[data_3["age_leval"] > 30]
	data_3 = data_3[data_3["age_leval"] <= 60]
	fugai.append(data_3)

	data_4 = data[data["sex"] == 1]
	data_4 = data_4[data_4["age_leval"] <= 30]
	fugai.append(data_4)

	data_5 = data[data["sex"] == 1]
	data_5 = data_5[data_5["age_leval"] > 30]
	data_5 = data_5[data_5["age_leval"] <= 55]
	fugai.append(data_5)

	data_6 = data[data["sex"] == 1]
	data_6 = data_6[data_6["age_leval"] > 55]
	fugai.append(data_6)

	for i in range(0,6):
		for column in hobby_list:
			print  len(fugai[i][fugai[i][column].notnull()]) * 1.0 / len(fugai[i]) * 100

def main():
	data = load_data("./data/sf/zh/new_data.csv")
	data['age_leval'] = data[['birth_date']].apply(lambda x: process_age(x[0]), axis=1)

	hobby_list = ["kyc_hobby", "app_hobby", "activity_hobby", "yl_hobby", "plane_hobby"]
	hobby = [""] * data.shape[0]
	for i in range(0, data.shape[0]):
		hobby[i] = ""
		for j in hobby_list:
			hobby[i]+=data[j][i]
			hobby[i]+='|'
	print hobby[1]
	data['hobby'] = hobby

	#customer_level
	data_1 = data[data["sex"] == 1]
	data_1 = data_1[data_1["age_leval"] <=30]
	# data_1 = data_1[data_1["age_leval"] <= 60]
	path = './final_plot/sex_1_age_leval_>30'

	# final_data = data[['sex', 'age_leval', 'fn_hobbies', 'birth_date']]
	# sex_1_age_leval30 = df_spit(final_data)
	# barplot(fn, "./final_plot/customer_level_2_fn.jpg", 10, sex_1_age_leval30)

	fn = map(lambda x: x.decode("utf-8"), df_to_list(data_1["fn_hobbies"]))
	barplot(fn, path+"_fn_hobbies.jpg", 10, data_1.shape[0])
	pieplot(fn, path + "_fn_hobbies_pie.jpg", 10, data_1.shape[0])

	ho= map(lambda x: x.decode("utf-8"), df_to_list(data_1["hobby"]))
	barplot(ho, path+"_hobby.jpg", 10, data_1.shape[0])
	pieplot(ho, path+"_hobby_pie.jpg", 10, data_1.shape[0])

if __name__ == "__main__":
	fugailv()
