#-*- coding: UTF-8 -*-
import csv
import sys
from fuzzywuzzy import fuzz
import re
import pandas as pd

def load_file(filename,oput_file_name):
    count=0
    with open(filename, 'r') as f:
        for line in f:
			line=line.split(',')
			count+=1
	print count

def output_result(ori_data, oput_file_name):
    output = open(oput_file_name, 'w')
    for index,item in enumerate(ori_data):
        output.write(','.join(item))


def main():
    load_file('apk_info.csv','./data/wandoujia_target.csv')
    # output_result(file_true, './file_true.csv')
    # output_result(file_false, './file_false')

if __name__ == '__main__':
    main()
