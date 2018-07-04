#encoding=utf-8
import csv
import scipy.stats as stats
import argparse
import pandas as pd
import time

def load_data(path):
    return pd.read_csv(path, error_bad_lines=False)
def des(dataname,savename):
    data = load_data(dataname)
    decrib = ['feature', '非空值个数', '覆盖率','非0个数','非0占比','列和','列和平均值']
    file0 = open(savename, 'wb+')  # 'wb'
    output = csv.writer(file0, dialect='excel')
    output.writerow(decrib)
    total = len(data)
    fea_list = data.columns

    print  len(fea_list)
    for m in range(1, len(fea_list)):
		colom = fea_list[m]
		try:
			colom = fea_list[m]
			print colom
			data_notnull = data[-data[colom].isnull()][colom]

			g_dist = sorted(data_notnull)
			lenth = len(g_dist)
			info = stats.describe(data_notnull)
			data_notnull_not_0=data_notnull[data_notnull!=0]
			leng0=len(data_notnull_not_0)
			sumdata=data_notnull.sum()
			listdes = [colom, lenth, float(int(info[0]) * 1.0 / total),leng0,leng0*1.0/lenth,sumdata,sumdata*1.0/lenth]
			output.writerow(listdes)
		except:
			pass
