#-*- coding: UTF-8 -*-
import csv
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import argparse
import numpy as np
import math
import os
parser = argparse.ArgumentParser()
parser.add_argument('--predict_name',type=str,default='test_multidim.txt',help='training data in csv format')
args = parser.parse_args()

def load_data(path):
    return pd.read_csv(path, error_bad_lines=False)


def find_max_cc():
    predict_name=args.predict_name
    # data=load_data(predict_name)
    # length=len(data)

    file=open(predict_name,'rb')
    myfile=[]
    for line in file.readlines():
        line=line.strip('\n').split('\t')
        for i in line:
            print int(i)

    maxtr = np.array(myfile)
    print maxtr
    m,n= maxtr.shape
    print m,n


    list=[1,2]
    sort_list_index=[]
    for line in list:
        sort_list_index.append(line[1])

    order = np.argsort(sort_list_index)
    final_sort_list=[]
    for index in order:
        final_sort_list.append(sort_list_index[index])

if __name__=="__main__":
    find_max_cc()
