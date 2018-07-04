#coding=utf-8
import pandas as pd
import csv
import scipy.stats as stats


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

    print len(fea_list)
    for m in range(3, len(fea_list)):
		colom = fea_list[m]
		try:
			if colom in ['comments','location','phone','calltime']:
				continue
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
			print "err"

def main():
	des('total_data/dianxiao_7','total_data/dianxiao_7_des.csv')
	des('total_data/dianxiao_10', 'total_data/dianxiao_10_des.csv')
if __name__ == '__main__':
    main()