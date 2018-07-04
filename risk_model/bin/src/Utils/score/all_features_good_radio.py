#-*- coding: UTF-8 -*-
import csv
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data_name',type=str,default='xxd_good_and_m7',help='training data in csv format')
parser.add_argument('--final',type=str,default='xxd_good_and_m7_finally.csv',help='training data in csv format')
args = parser.parse_args()
def load_data(path):
    return pd.read_csv(path, error_bad_lines=False)
def des():
    final=args.final
    final = open(final, 'wb+')  # 'wb'
    output = csv.writer(final, dialect='excel')

    dataname = args.data_name
    file = open(dataname, 'rb')
    number = 0

    for row_data1 in file.readlines():

        row_data = row_data1.replace('NULL', '0.0')
        row_data = row_data.strip()
        row_data = row_data.split(",")

        row_data1 = row_data1.strip()
        row_data1 = row_data1.split(",")

        num=0
        nump=0
        maxnump=0
        number += 1
        try:
            if row_data1[38] == 'NULL' and row_data1[39] == 'NULL' and row_data1[40] == 'NULL' and row_data1[
                    41] == 'NULL' and row_data1[42] == 'NULL' and row_data1[43] == 'NULL':
                cha = "NULL"
                mean1="NULL"

            else:
                cha = max(float(row_data[38]),float(row_data[39]),float(row_data[40]),float(row_data[41]),float(row_data[42]),float(row_data[43]))- \
                      min(float(row_data[38]),float(row_data[39]),float(row_data[40]),float(row_data[41]),float(row_data[42]),float(row_data[43]))
                mean1=(float(row_data[38])+float(row_data[39])+float(row_data[40])+float(row_data[41])+float(row_data[42])+float(row_data[43]))/6

            if row_data1[38] == 'NULL' and row_data1[39] == 'NULL' and row_data1[40] == 'NULL' and row_data1[
                41] == 'NULL' and row_data1[42] == 'NULL' and row_data1[43] == 'NULL':
                nump = "NULL"
                maxnump = "NULL"
                num = "NULL"

            else:

                for i in range(5):
                    if row_data1[i + 38] == "0.0" or row_data1[i + 38] == '0.0' or row_data1[i + 38] == 'NULL':
                        num += 1
                        nump += 1
                    if row_data1[i + 39] !="0.0" and row_data1[i + 39] != '0.0' and row_data1[i + 39] != 'NULL' and i+39<44:
                        if maxnump<nump:
                            maxnump=nump
                        else:
                            nump=0

        except:
            print "err"
            cha="gap"
            mean1="mean"
            num="card_interrupt"
            maxnump="max_continuous_interrupt_month"

        output.writerow([
            row_data1[6],row_data[1],row_data[2],row_data[3],row_data[4],row_data[5],num,cha,mean1,maxnump,row_data1[7],row_data1[8],row_data1[9],row_data1[10],row_data1[11],row_data1[12],row_data1[13],
            row_data1[14],row_data1[15],row_data1[16],row_data1[17],row_data1[18],row_data1[19],row_data1[20],row_data1[21],row_data1[22],row_data1[23],
            row_data1[24],row_data1[25],row_data1[26],row_data1[27],row_data1[28],row_data1[29],row_data1[30],
            row_data1[31],row_data1[32],row_data1[33],row_data1[34],row_data1[35],row_data1[36],row_data1[37],row_data1[38],row_data1[39],row_data1[40],
            row_data1[41], row_data1[42], row_data1[43], row_data1[44], row_data1[45], row_data1[46], row_data1[47],
            row_data1[48], row_data1[49], row_data1[50],
            row_data1[51], row_data1[52], row_data1[53], row_data1[54], row_data1[55], row_data1[56], row_data1[57],
            row_data1[58], row_data1[59], row_data1[60],
            row_data1[61], row_data1[62], row_data1[63], row_data1[64], row_data1[65], row_data1[66], row_data1[67],
            row_data1[68], row_data1[69], row_data1[70],
            row_data1[71], row_data1[72], row_data1[73], row_data1[74], row_data1[75], row_data1[76], row_data1[77],
            row_data1[78], row_data1[79], row_data1[80],row_data1[81],row_data1[82],row_data1[83], row_data1[84],
            row_data1[85],row_data1[86],row_data1[87]])

    print number

if __name__ == '__main__':
    des()






