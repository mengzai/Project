#coding=utf-8
import csv
FORMAT = "%Y-%m-%d"
ENABLE_DEBUG = False
cutoff_date_str="2016-11-30"

import sys
import datetime
#转出:to_csv
def output_data(names_value, cutoff_date,output_file_name):
    if output_file_name[-3:] == "csv":
        with open(output_file_name, 'w') as file0:
            output = csv.writer(file0, dialect='excel')
            for item in names_value:
                item.append(cutoff_date)
                output.writerow(item)

    elif output_file_name[-3:] == "txt":
        with open(output_file_name, 'w') as output:
            for item in names_value:
                item.append(cutoff_date)
                for i in range(0,len(item)):
                    output.write('%s\t' % (item[i]))
                output.write('\n')
    else:
        print "此输出文件非txt 及csv,请以 txt or csv  结尾"


def load_file(filename):
    if filename[-3:]=="txt":
        with open(filename, 'r') as data_file:
            data = []
            for line in data_file.readlines():
                line = line.strip('\n').split('\t')
                data.append(line)
            return data
    if filename[-3:] == "csv":
        with open(filename, 'r') as data_file:
            data = []
            for line in data_file.readlines():
                line = line.strip('\r\n').split(',')
                data.append(line)
            return data
    else:
        print "此输入文件非txt 及csv,请以 txt or csv  结尾"
def main():
	score_txt=load_file('final_score1')
	cutoff_date = datetime.datetime.strptime(cutoff_date_str, FORMAT)
	cutoff_date = cutoff_date.strftime(FORMAT)
	print cutoff_date
	output_data(score_txt,cutoff_date,'final_score_mysql.txt')


if __name__ == "__main__":
    main()


