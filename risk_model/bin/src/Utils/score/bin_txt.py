# coding:utf-8
# ! /bin/python
import os
import sys
import os.path
import pickle
import struct
import argparse
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--txt_name', type=str, default='model.txt', help='type data txt')
parser.add_argument('--bin_name', type=str, default='model.sn', help='type data bin')
args = parser.parse_args()

"""
参数设置:
1:--txt_name :原本的txt文件名
2:--bin_name :转出的bin文件
"""

def change_txt_to_bin(binnames, txtname):

    fileNew = open(binnames, 'rb')
    for i in range(1,273):
        data_id = struct.unpack("f", fileNew.read(4*i))
        print data_id

if __name__ == "__main__":
    dirnames = args.txt_name
    filename = args.bin_name
    change_txt_to_bin(filename, dirnames)
