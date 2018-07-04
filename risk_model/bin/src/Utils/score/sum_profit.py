#-*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import pickle
import math
import datetime

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pre_name',type=str,default='logistic_regression/test_result.txt',help='training data in csv format')
parser.add_argument('--data_name',type=str,default='logistic_regression/GENDER_test_24_1',help='training data in csv format')
args = parser.parse_args()

def load_data(path):
    return pd.read_csv(path, error_bad_lines=False)



def max_gap():
    pre_name = args.pre_name
    data_name = args.data_name

    pre_name0=load_data(pre_name)
    data_name0=load_data(data_name)
    print len(pre_name0),len(data_name0)


    pre = open(pre_name)

    prelist = []
    for line in pre.readlines():
        line = line.strip()
        line = line.split("\t")
        if float(line[0]) == 1 and float(line[2]) < 0.574116:
            prelist.append(1)
        elif float(line[0]) == 0 and float(line[2]) < 0.574116:
            prelist.append(0)
        else:
            prelist.append(2)

    data = open(data_name)
    num = 0
    total = 0
    loss = 0
    no_loss = 0
    num_bad = 0
    num_good = 0
    for lines in data.readlines():
        if num == 0:
            num += 1
            continue
        lines = lines.split(",")
        total += float(lines[-3])
        # if prelist[num]!=2:
        # print lines[-3], lines[-4], prelist[num-1]

        if prelist[num - 1] == 1:
            loss += float(lines[-3])
            num_bad += 1
        if prelist[num - 1] == 0:
            no_loss += float(lines[-3])
            num_good += 1
        num += 1
    loss = abs(loss)
    no_loss = abs(no_loss)
    print "共有人数", num, "总收益",total
    print "把好人误判为坏人的人数 : ",num_bad, "把好人误判为坏人的损失 : ",loss
    print "把坏人判为坏人的人数 : ", num_good,"把坏人判为坏人的少损失即收益 : ", no_loss
    print "收益百分比", (no_loss - loss) / total

if __name__ == '__main__':
    max_gap()
