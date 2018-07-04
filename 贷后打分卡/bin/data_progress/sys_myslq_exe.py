#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import os
import datetime
import time
import re
import subprocess
import shlex

reload(sys)
sys.setdefaultencoding('utf8')

def get_argv(args):
    if len(args) == 2:
        return True
    else:
        print "Please input etl sql file path . then re-run the script."
        return False
def job_exc(job_file_name):

    job_file= job_file_name

    cmd = "hive -f {sql} ".format(sql=job_file)

    print "===========================" + cmd
    flag= 0
    flag = subprocess.call(shlex.split(cmd.encode('UTF-8')))
    print "flag :", flag

    return flag

if  __name__ == "__main__":
    d=datetime.datetime.now()
    week_day=d.weekday()
    print week_day
    if int(week_day)==6:
    	if not get_argv(sys.argv):
       		sys.exit(1)
    	job_file_name=sys.argv[1]
    	flag = job_exc(job_file_name)
    	exit(flag)
    else:
	print "this is not Tuesday"
