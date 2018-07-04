#-*- coding: UTF-8 -*-
import csv
import argparse
import pandas as pd

"""
A:true ok   predict ok
B:true ok   predict M7
C:true M7   predict ok
D:true M7   predict M7
"""

parser = argparse.ArgumentParser()
parser.add_argument('--data_name',type=str,default='xxd_good_and_m7',help='training data in csv format')
parser.add_argument('--xxd_good_and_m7_new',type=str,default='xxd_good_and_m7_new',help='training data in csv format')



args = parser.parse_args()
def load_data(path):
    return pd.read_csv(path, error_bad_lines=False)
def des():

    data_name=args.data_name
    xxd_good_and_m7_new = args.xxd_good_and_m7_new


    data=load_data(data_name)
    fea_list = data.columns

    filexxd_good_and_m7_new = open(xxd_good_and_m7_new, 'wb+')  # 'wb'
    outputxxd_good_and_m7_new = csv.writer(filexxd_good_and_m7_new, dialect='excel')
    outputxxd_good_and_m7_new.writerow(fea_list)

    transport_id=[]
    file1 = open("transport_id_new")
    for line in file1.readlines():
        try:
            line = line.strip()
            line = line.split(" ")
            transport_id.append(int(float(line[0])))
        except:
            print "err1"

    dataname = args.data_name
    file = open(dataname, 'rb')
    number = 0

    for row_data in file.readlines():
        number+1
        row_data = row_data.strip()
        row_data = row_data.split(",")
        try:
            if int(float(row_data[2])) in transport_id:
                # print int(float(row_data[2]))
                outputxxd_good_and_m7_new.writerow(row_data)
        except:
            print "err2"
    print number


if __name__ == '__main__':
    des()





