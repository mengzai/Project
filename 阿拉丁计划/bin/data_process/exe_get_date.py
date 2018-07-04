#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import os
import datetime
import time
import re
import subprocess
import shlex
import pandas  as pd


reload(sys)
sys.setdefaultencoding('utf8')



def datetostr(date):
    return   str(date)[0:10]

def getDaysByNum(start_date,end_date):
	num=(end_date - start_date).days
	oneday=datetime.timedelta(days=1)
	li=[]
	for i in range(0,num):
		start_date=start_date+oneday
		li.append(datetostr(start_date))
	return li

def get_data():
	train_start_date =  datetime.datetime.strptime('2016-04-01',"%Y-%m-%d")
	train_end_date = datetime.datetime.strptime('2016-10-01',"%Y-%m-%d")
	test_start_date = datetime.datetime.strptime( '2017-01-01',"%Y-%m-%d")
	test_end_date = datetime.datetime.strptime('2017-04-01',"%Y-%m-%d")
	train_data_list=getDaysByNum(train_start_date,train_end_date)
	test_data_list = getDaysByNum(test_start_date, test_end_date)
	return train_data_list,test_data_list

def job_exc(job_file_name):
	job_file = job_file_name
	train_data_list, test_data_list=get_data()
	for data_date in train_data_list:
		print data_date
		cmd = "hive -hiveconf data_date={data_dt} -v -f  {sql}".format(sql=job_file, data_dt=data_date)


		print "===========================" + cmd
		flag = 0
		flag = subprocess.call(shlex.split(cmd.encode('UTF-8')))
		print "flag :", flag

	return flag


if __name__ == "__main__":
	job_file_name = sys.argv[0]
	flag = job_exc(job_file_name)
