# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import csv
import matplotlib.pyplot as plt
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
			hobb=line[31].decode("gbk").split('|')
			cnt_count.append(hobb)
			ecif_id_air_count.setdefault(line[0],[]).extend(hobb)
			count+=1
	print count
	return cnt_count,ecif_id_air_count

def output_result(ori_data, oput_file_name):
    output = open(oput_file_name, 'w')
    for index,item in enumerate(ori_data):
        output.write(','.join(item))

def  count_cnt(cnt_count):
	cnt=[]
	set_list=list(set(cnt_count))
	print set_list
	for m  in set_list:
		cnt.append(cnt_count.count(m))
	print cnt

	return set_list,cnt

def plot_bar(set_list, cnt):
	labels=['1次','2次','3次','4次']
	fig = plt.figure(1, figsize=(10, 8))
	ax = fig.add_subplot(111)
	ax.bar(set_list, cnt, 0.3, color='b', label='bar')
	ax.set_xticks(set_list)
	ax.set_xticklabels(labels)
	plt.show()
	fig.savefig('fenbu' + '.png')

def hanzi_count():
	output = open('./data/sf/zh/kyc.csv', 'w')
	cnt_count,ecif_id_air_count = load_file_trval_dict('./data/sf/zh/MGM_vv5.csv')

	ecif_id_air_count = load_file_trval_dict("./data/sf/zh/MGM_feature_app.csv")
	print ecif_id_air_count
	chanpin = []
	for m in ecif_id_air_count:
		ecif_id=[m]
		ecif = []
		ecif.extend(list(set(ecif_id_air_count[m])))
		chanpin.extend(list(set(ecif_id_air_count[m])))
		if '其它' in ecif:
			ecif.remove('其它')
		output.write(','.join(ecif_id))
		output.write(',')
		output.write('|'.join(ecif))
		output.write('\n')

	file = open('./data/sf/zh/kyc_tongji.csv', 'w')
	output1 = csv.writer(file, dialect='excel')
	for p in list(set(chanpin)):
		print p, chanpin.count(p)
		output1.writerow([p, chanpin.count(p)])

def main():
	hanzi_count()
	# cnt_count,ecif_id_air_count=load_file_trval_dict('./data/sf/zh/MGM_vv5.csv')
	# set_list, cnt=count_cnt(cnt_count)
	# plot_bar(set_list, cnt)
if __name__ == '__main__':
    main()