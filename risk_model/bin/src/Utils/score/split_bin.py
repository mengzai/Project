import pandas as pd
import math
import numpy
import pickle


def split_bin(dataframe, pickle_path):

    """
        This function split bins for each feature in the dataframe.
        INPUT:
          1)dataframe: dataframe format. The first column must be label!!
          2)pickle_path: location to store the output
        OUTPUT:
          pickle: a dictionary:
            key is feature name,
            value is a list which contains three parts: bin points(list), logodds for each bin(list), logodds for null(if have null values)
    """


    fea_list = dataframe.columns#feature list, the first one is label
    output={}#output dictionary

    for k in range(1, len(fea_list)):#process one column each time
        col_name = fea_list[k]
        data = dataframe[[col_name, 'label']]
        data_notnull = data[-data[col_name].isnull()]
        #sorted_col = sorted(data_notnull[col_name])#sort the column value asc
        index = numpy.argsort(data_notnull[col_name])
        sorted_col=data_notnull.iloc[index, 0]
        sorted_col=list(sorted_col)
        label = data_notnull.iloc[index, 1]#sort label in the same order as column value
        label = list(label)

        ##########################################  bin_point  #####################################################
        # set the maximum number of bins
        num_bin = 10
        #minimum number of points in each bin
        min_num =int(len(data_notnull) * 1.0 / num_bin)
        # bin_point is bin points, the first bin point is the minimum point of the column value
        bin_point = [sorted_col[0]]
        # index1 is the location of bin point
        index1 = [0]
        i = 0

        while i < len(data_notnull):
            if (len(data_notnull) - i > min_num):
                i = i + min_num
                tmp = sorted_col[i]
                for j in range(i + 1, len(data_notnull)):
                    if (sorted_col[j] == tmp):
                        j = j + 1
                    else:
                        tmp = sorted_col[j - 1]
                        i = j - 1
                        index1.append(j - 1)
                        bin_point.append(tmp)
                        break
            else:
                break
        # if the last bin is too small, combine it with the previous one
        if (len(data_notnull) - 1 - index1[-1] < min_num and index1[-1] != len(data_notnull) - 1):
            bin_point.pop(-1)
            index1.pop(-1)
        # add the last point to the bin_point if the last point is not in the binpoint
        if (index1[-1] != len(data_notnull) - 1):
            index1.append(len(data_notnull) - 1)
            bin_point.append(sorted_col[-1])

        bin_point[0]=bin_point[0]-0.5
        bin_point[-1]=bin_point[-1]+0.5

        for i in range(1,len(bin_point)-1):
            bin_point[i]=(sorted_col[index1[i]]+sorted_col[index1[i]+1])*1.0/2

        ################################################## calc odds ##########################################################
        index1[0] = index1[0] - 1
        # group of value
        group = [sorted_col[index1[i] + 1: index1[i + 1] + 1] for i in range(0, len(index1) - 1)]
        # group of label
        group_label = [label[index1[i] + 1: index1[i + 1] + 1] for i in range(0, len(index1) - 1)]
        index1[0] = index1[0] + 1

        bin_N_good = [[len(group[i]), sum(group_label[i])] for i in range(0, len(index1) - 1)]
        logodds=[]

        for i in range(0, len(index1) - 1):
            if(bin_N_good[i][1]==bin_N_good[i][0]):#if the bin only has good
                cur_odds = 9
            elif(bin_N_good[i][1]==0): #if the bin only has bad
                cur_odds=-9
            else:
                odds_origin=math.log(bin_N_good[i][1] * 1.0 / (bin_N_good[i][0] - bin_N_good[i][1]))
                cur_odds=min(max(-9, odds_origin), 9)
            logodds.append(cur_odds)

        output[col_name]=[bin_point,logodds]

        if (sum(data[col_name].isnull()) > 0): #if the column has null value
            null_N_good = [sum(data[col_name].isnull()),
                           sum(map(lambda x, y: x & y, data[col_name].isnull(), data['label'] == 1))]
            if(null_N_good[0]==null_N_good[1]):#if null only has good
                null_logodds=9
            elif(null_N_good[1]==0):#if null only has bad
                null_logodds=-9
            else:
                null_logodds_origin = math.log(null_N_good[1] * 1.0 / (null_N_good[0] - null_N_good[1]))
                null_logodds=min(max(-9, null_logodds_origin), 9)
            output[col_name].append(null_logodds)


    ################################################## output pickle ##########################################################

    Path=pickle_path+'/data_all/dictionary.pkl'
    file = open(Path, 'wb')
    pickle.dump(output, file)
    file.close()


index = numpy.argsort(delay['delay'])
sorted_col = list(delay.iloc[index, 0])
label = list(delay.iloc[index, 1])  # sort label in the same order as column value
label = list(label)

while i < len(index):
    if (len(data_notnull) - i > min_num):
        i = i + min_num
        tmp = sorted_col[i]
        for j in range(i + 1, len(data_notnull)):
            if (sorted_col[j] == tmp):
                j = j + 1
            else:
                tmp = sorted_col[j - 1]
                i = j - 1
                index1.append(j - 1)
                bin_point.append(tmp)
                break
    else:
        break

for i in range()