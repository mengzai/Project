#coding=utf-8
import pandas as pd
import snailseg
import csv
import scipy.stats as stats
# words = snailseg.cut("电话接通,且没有人说话")
# print words


def load_data(path):
    return pd.read_csv(path, error_bad_lines=False)


def commert():
	good=[
'注册',
'门店',
'费率',
'考虑一下',
'营业部',
'考虑',
'资金',
'做生意',
'结清',
'周转',
'精英',
'进件',
'营业执照',
'W',
'公司',
'太高',
'按揭',
'用钱',
'一个月',
'贷过',
'开店',
'批核',
'审核',
'展期',
'减免',
'借到',
'不划算',
'抵押',
'越多越好',
'征信',
'注册',
'费率',
'办理',
'买房',
'预约',
'愿意',
'回访',
'办理',
'商量',
'填表',
'网查' ]
	bad=[
		'拒接',
		'挂机',
		'空号',
		'未接通',
		'占线',
		'未通',
		'打错',
		'不办',
		'暂不需',
		'不行',
		'没空',
		'拒绝',
		'拦截',
		'拉黑',
		'呆账',
		'有误',
		'失败',
		'无应答',
		'拒绝',
		'虚假'
	]


def des_good_bad(bad_list,bad_set,good_list,good_set,g,b):
	file0 = open('total_data/word_good_count.txt', 'wb')
	output0 = csv.writer(file0, dialect='excel')

	file1 = open('total_data/word_bad_count.txt', 'wb')
	output1 = csv.writer(file1, dialect='excel')

	bad_count={}
	for line in bad_set:
		bad_count[line]=bad_list.count(line)*1.0/b

	print b

	good_count = {}
	for line1 in good_set:
		good_count[line1] = good_list.count(line1)*1.0/g
	print g
	print 2
	for key ,val in good_count.items():
		output0.writerow([key ,val])
	for key, val in bad_count.items():
		output1.writerow([key, val])


def Open_data():
	line_data=open('total_data/dianxiao_7')
	file0=open('word_good.txt','wb')
	output0 = csv.writer(file0, dialect='excel')

	file1 = open('word_bad.txt', 'wb')
	output1 = csv.writer(file1, dialect='excel')

	good_list=[]

	bad_list = []
	g,b=0,0

	for line in line_data.readlines():
		line=line.split(",")
		words = snailseg.cut(line[13])
		word_list = []

		words=list(set(words))

		for i in words:
			word_list.append(i.encode('utf-8'))
			if line[0] == '0':
				bad_list.append(i.encode('utf-8'))
			else:
				good_list.append(i.encode('utf-8'))

		if line[0]=='0':
			b+=1
			output1.writerow(list(set(word_list)))
		else:
			g+=1
			output0.writerow(list(set(word_list)))
	bad_set=set(bad_list)
	good_set = set(good_list)
	print g,b

	print bad_list.count('无人'),bad_list.count('无人')*1.0/b
	return bad_list,bad_set,good_list,good_set,g,b

def Sort():
	file0 = open('total_data/word_good_count.txt', 'rb')
	file1 = open('total_data/word_bad_count.txt', 'rb')


	filea = open('total_data/word_good_sort.csv', 'wb')
	outputa = csv.writer(filea, dialect='excel')

	fileb = open('total_data/word_bad_sort.csv', 'wb')
	outputb = csv.writer(fileb, dialect='excel')

	good_dict={}
	for line in file0.readlines():
		line=line.split(',')
		good_dict[line[0]]=float(line[1])
	bad_dict = {}
	for line in file1.readlines():

		line = line.split(',')
		if float(line[1])>1:
			print line[0],line[1]
		bad_dict[line[0]] = float(line[1])
	good=sorted(good_dict.iteritems(), key=lambda d:d[1],reverse=True)
	bad=sorted(bad_dict.iteritems(), key=lambda d:d[1],reverse=True)

	# print good
	list_set=set()
	for line  in good:
		# line=list[line]
		# line = line.split(',')
		list_set.add(line[0])
		outputa.writerow(line)
	for line  in bad:
		# line = list[line]
		# line = line.split(',')
		list_set.add(line[0])
		outputb.writerow(line)
	return list_set,good_dict,bad_dict


def diff(list_set,good,bad):
	filea = open('total_data/differ.csv', 'wb')
	outputa = csv.writer(filea, dialect='excel')

	# print list_set
	differ={}
	for i in list_set:
		if good.has_key(i) :
			if bad.has_key(i):
				differ[i]=[good[i]-bad[i],good[i],bad[i]]
			else:
				differ[i]=[good[i]-0,good[i],0]
		else:
			if bad.has_key(i):
				differ[i]=[0 - bad[i],0,bad[i]]
			else:
				differ[i]=0
	# print differ

	for key, val in differ.items():

		outputa.writerow([key, val[0],val[1],val[2]])


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
def main():
	bad_list, bad_set, good_list, good_set,g,b=Open_data()
	print 1
	des_good_bad(bad_list, bad_set, good_list, good_set,g,b)
	print 2
	list_set,good,bad=Sort()
	print 3
	diff(list_set,good,bad)
if __name__ == '__main__':
    main()