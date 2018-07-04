# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from collections import Counter
import datetime
FORMAT = "%Y-%m-%d"
def df_to_list(df):
    df = df.tolist()
    res = []
    for each in df:
        vals = [k for k in str(each).split("|") if not pd.isnull(k) and k != 'nan' and k!= 'None']
        res += vals
    return res

# barplot(a)
def barplot(a, savename='./final_plot/sex_1_age_leval30', most_common=10):
    cntdict = Counter(a)
    cnttuple = cntdict.most_common(most_common)
    index = range(len(cnttuple))
    values = [val[1] for val in cnttuple]
    xticks = [val[0].decode('utf-8') for val in cnttuple]
    plt.bar(index, values)

    plt.title(savename.split('.')[0])
    plt.xticks(index, xticks)

    # plt.xlabel(u"横坐标xlabel", fontproperties=xticks)
    if savename:
        plt.savefig(savename + '.png')

	plt.show()
    plt.close()

def pieplot(a, savename='pie.png', most_common=10):
    cntdict = Counter(a)
    cnttuple = cntdict.most_common(most_common)
    index = range(len(cnttuple))
    values = [val[1] for val in cnttuple]
    xticks = [val[0] for val in cnttuple]
    plt.pie(values, labels=xticks)
    plt.title(savename.split('.')[0])
    # plt.xticks(index, xticks)
    plt.show()
    if savename:
        plt.savefig(savename)
    plt.close()

def load_data(path):
    return pd.read_csv(path, error_bad_lines=False)

def process_age(x):
    cutoff_now=datetime.datetime.now()

    cutoff_x = datetime.datetime.strptime(x,FORMAT)
    return int (((cutoff_now-cutoff_x).days)/365)


def  df_spit(final_data):
	sex_1_age_leval30 = final_data[(final_data["sex"] ==1) &  (final_data['age_leval'] <= 60)&  (final_data['age_leval'] > 30)]
	return sex_1_age_leval30['fn_hobbies']

def final_plot(data):
	a=df_to_list(data)
	b= Counter(a)
	print b.most_common(3)
	barplot(a)

if __name__ == '__main__':
    data=load_data('./data/sf/zh/new_data.csv')
    data['age_leval'] = data[['birth_date']].apply(lambda x: process_age(x[0]), axis=1)
    final_data= data[['sex','age_leval','fn_hobbies','birth_date']]
    sex_1_age_leval30=df_spit(final_data)
    final_plot(sex_1_age_leval30)




