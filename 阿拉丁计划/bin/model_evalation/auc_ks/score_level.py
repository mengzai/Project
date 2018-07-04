# encoding=utf-8
import csv
import pandas as pd
import math


def read_data(Path):
	day_data = pd.read_csv(Path, sep=',', quoting=csv.QUOTE_NONE, low_memory=False)
	return day_data


def output_table(data, Path):
	f = open(Path, "w")
	data.to_csv(Path, sep=',', index=False, header=False)
	f.close()

def trans_to_score(predicts):
	purity_score = []
	for predict in predicts:
		if (predict / (1 - predict)) > 1 * (10 ** (-8)):
			value = 1.0 / (1.0 + math.e ** (-1.07201816474 * math.log(predict / (1 - predict)) + 3.21423064832))
			purity_score.append(value)
		else:
			value = 1 * (10 ** (-8))
			purity_score.append(value)
	return purity_score

def get_fenwei(purity_score, level_nums):
	fenwei = []
	scores = sorted(purity_score)
	length = len(scores)
	bin_length = int(length / level_nums)
	for i in range(level_nums + 1):
		if i == 0:
			fenwei.append("%.14f" % 0.0)
		elif i != level_nums and i != 0:
			pos = i * bin_length
			fenwei.append("%.14f" % scores[pos])
		else:
			fenwei.append("%.14f" % 1.1)
	return fenwei

def get_fenji(purity_score, level_nums):
	fenwei = get_fenwei(purity_score, level_nums)
	grade = []
	for score in purity_score:
		for i in range(len(fenwei) - 1):
			if (float(score) >= float(fenwei[level_nums - (i + 1)]) and float(score) < float(fenwei[level_nums - i])):
				grade.append(i + 1)
				break
	return grade

def grade_count(good,all_people,path):
	grade_data=open('./data/dianxiao_model_score')
	leval_data={}
	for line in grade_data.readlines():
		line=line.strip('\n')
		leval=int(float(line.split(',')[0]))
		label=int(float(line.split(',')[2]))
		leval_data.setdefault(leval, []).append(label)

	data_file = open(path, 'w')
	output = csv.writer(data_file, dialect='excel')
	output.writerow(['leval','good_num','good_percent','all_people'])
	for le  in leval_data:
		print le,len(leval_data[le]),sum(leval_data[le]),sum(leval_data[le])*1.0/good,all_people
		output.writerow([le,len(leval_data[le]),sum(leval_data[le]),sum(leval_data[le])*1.0/good,all_people])

def main():
	data = read_data(datadir + 'dianxiao_metrics_test.txt')
	score_grade = list(data['probability'])
	score = [int(round(x * 10000)) for x in score_grade]
	data['score'] = score

	data['grade'] = get_fenji(score_grade, 10)  # 分20级
	data = data[['grade', 'score','label']]
	good= len(data[data['label']==1.0] )
	all_people=len(data)
	grade_count(good,all_people,datadir + 'dianxiao_model_leval')
	output_table(data, datadir + 'dianxiao_model_score')

if __name__ == '__main__':
	datadir = './data/'
	main()