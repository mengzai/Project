import sys

import collections

"""

Input Format: mortgagor_id,latest_month_income,n1,n2,n3,n4,n5

"""

print 1
number = 0
for row_data1 in sys.stdin:
    row_data = row_data1.replace('NULL', '0.0')

    row_data = row_data.strip().split('\t')

    row_data1 = row_data1.strip().split('\t')

    num = 0
    nump = 0
    maxnump = 0
    number += 1

    try:
        if row_data1[1] == 'NULL' and row_data1[2] == 'NULL' and row_data1[3] == 'NULL' and row_data1[
            4] == 'NULL' and row_data1[5] == 'NULL' and row_data1[6] == 'NULL':
            cha = "NULL"
            mean1 = "NULL"

        else:
            cha = max(float(row_data[1]), float(row_data[2]), float(row_data[3]), float(row_data[4]),
                      float(row_data[5]), float(row_data[6])) - \
                  min(float(row_data[1]), float(row_data[2]), float(row_data[3]), float(row_data[4]),
                      float(row_data[5]), float(row_data[6]))
            mean1 = (float(row_data[1]) + float(row_data[2]) + float(row_data[3]) + float(row_data[4]) + float(
                row_data[5]) + float(row_data[6])) / 6

        if row_data1[1] == 'NULL' and row_data1[2] == 'NULL' and row_data1[3] == 'NULL' and row_data1[
            4] == 'NULL' and row_data1[5] == 'NULL' and row_data1[6] == 'NULL':
            nump = "NULL"
            maxnump = "NULL"
            num = "NULL"

        else:

            for i in range(6):
                if row_data1[i + 1] == "0.0" or row_data1[i + 1] == '0.0' or row_data1[i + 1] == 'NULL':
                    num += 1
                    nump += 1
                if row_data1[i + 2] != "0.0" and row_data1[i + 2] != '0.0' and row_data1[i + 2] != 'NULL' and i != 6:
                    if maxnump < nump:
                        maxnump = nump
                    else:
                        nump = 0

    except:
        print "err"
    print '\t'.join([num,cha,mean1,maxnump])





#
# import sys
# for row_data1 in sys.stdin:
#     row_data = row_data.strip()
#     row_data = row_data.split("\t")
#     print '\t'.join(row_data[0],row_data[1],row_data[2],row_data[3])
#
#
# output_row = collections.defaultdict(list)
# apply_id_set = set()
#
#
#
# for row in sys.stdin:
#     row_list = row.strip().split('\t')
#     if len(apply_id_set) != 0 and row_list[0] not in apply_id_set:
#         for row in output_row.values():
#             print '\t'.join(row)
#     output_row = collections.defaultdict(list)
#     apply_id_set = set()
#     apply_id_set.add(row_list[0])
#     if row_list[2] not in output_row:
#         output_row[row_list[2]] = row_list
#     else:
#         if float(row_list[1]) > float(output_row[row_list[2]][1]):
#             output_row[row_list[2]] = row_list
#
# for row in output_row.values():
#     print '\t'.join(row)