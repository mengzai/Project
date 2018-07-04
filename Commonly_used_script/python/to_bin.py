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
parser.add_argument('--txt_name', type=str, default='lr/test_multidim.txt', help='type data txt')
parser.add_argument('--bin_name', type=str, default='lr/test_multidim.bin', help='type data bin')
args = parser.parse_args()

"""
参数设置:
1:--txt_name :原本的txt文件名
2:--bin_name :转出的bin文件
"""

def change_txt_to_bin(dirnames, filename):
    num = 0
    anamefile = open(dirnames, 'r')
    for aline in anamefile.readlines():
        aline = aline.strip('\n')
        acurLine = aline.split('\t')
        num += 1
        dim = (len(acurLine) - 1)
    anamefile.close()

    file = open(dirnames, 'r')
    fileNew = open(filename, 'wb')
    parsedata = struct.pack("i", num)
    fileNew.write(parsedata)
    parsedata = struct.pack("i", dim)
    fileNew.write(parsedata)

    lines = file.readlines()
    for line in lines:
        line = line.strip('\n')
        curLine = line.split('\t')
        for i in range(len(curLine)):
            if len(curLine[i]) == 0:
                continue
            if i == 0:
                parsedata = struct.pack("i", int(curLine[i]))
            else:
                parsedata = struct.pack("f", float(curLine[i]))
        fileNew.write(parsedata)
    fileNew.close()
    file.close()

if __name__ == "__main__":
    dirnames = args.txt_name
    filename = args.bin_name
    change_txt_to_bin(dirnames, filename)
