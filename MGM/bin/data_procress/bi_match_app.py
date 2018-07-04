# encoding=utf8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

import math
def load_file_trval_dict(filename):
    count=0
    ecif_id_air_count={}
    cnt_count=[]
    cnt=0
    with open(filename, 'r') as f:
        for line in f:
			line.strip('\r\n')
			line=line.split(',')
			if count==0:
				count += 1
				continue
			# cnt_count.append(int(line[1]))
			if line[2]!='':
				ecif_id_air_count.setdefault(line[2],[]).append(line[4])
			count+=1
	print count
	return ecif_id_air_count

def df_ana(path):
	data=pd.read_csv(path, error_bad_lines=False)
	cat=data['cat']
	cat.fillna('missing')
	count= cat.value_counts()
	count.plot(kind='barh')
	plt.show()


def main():
	df_ana('./data/sf/zh/MGM_feature_app.csv')
	output = open('./data/sf/zh/app.csv', 'w')
	ecif_id_air_count = load_file_trval_dict("./data/sf/zh/MGM_feature_app.csv")

	chanpin=[]
	for m in  ecif_id_air_count:
		ecif_id = [m]
		ecif=[]

		ecif.extend(list(set(ecif_id_air_count[m])))
		chanpin.extend(list(set(ecif_id_air_count[m])))
		output.write(','.join(ecif_id))
		output.write(',')
		output.write('|'.join(ecif))
		output.write('\n')

	file = open('./data/sf/zh/app_tongji.csv', 'w')
	output1 = csv.writer(file, dialect='excel')
	for p in list(set(chanpin)):
		print p,chanpin.count(p)
		output1.writerow([p,chanpin.count(p)])

if __name__ == '__main__':
    main()