#-*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import pickle
import math
import datetime
import  numpy
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_name',type=str,default='data_all/xxd_version1',help='training data in csv format')
parser.add_argument('--logistic_regression',type=str,default='logistic_regression',help='training data in csv format')
parser.add_argument('--term',type=str,default="version",help='training data in csv format')
args = parser.parse_args()

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

def split_bin_su(dataframe_my, pickle_path,SP,num_bin = 10):

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

    import pandas as pd
    import math
    import numpy
    import pickle


    fea_list = dataframe_my.columns#feature list, the first one is label
    output={}#output dictionary

    for k in range(0, len(fea_list) - 2):#process one column each time
        col_name = fea_list[k]
        # print col_name

        data = dataframe_my[[col_name, 'label']]
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


    # Path = pickle_path + "/" + SP + '_dictionary1.pkl'
    # file = open(Path, 'wb')
    # pickle.dump(output, file)
    # file.close()
    return output


def split_bin(dataframe_my, pickle_path,SP,num_bin = 10):
    fea_list = dataframe_my.columns
    output={}

    for k in range(0, len(fea_list)-2):
        col_name = fea_list[k]
        print col_name
        data = dataframe_my[[col_name, 'label']]
        data_notnull = data[-data[col_name].isnull()]

        #将数据从小到大进行排序,并将label排序   返回数据从小到大的索引值,并根据索引值得到数据从小到大的排序值同理将lable进行排序,并转化为list
        index = np.argsort(data_notnull[col_name])
        sorted_col=list(data_notnull.iloc[index, 0])
        sorted_col_set=sorted(list(set(sorted_col)))
        label = list(data_notnull.iloc[index, 1])#sort label in the same order as column value


        ##########################################  bin_point  #####################################################
        #根据num_bin  得到每个bin里面最少有多少个值,并将第一个bin_point为开始值
        min_num =int(len(data_notnull) * 1.0 / num_bin)
        bin_point = [sorted_col[0]]
        index1 = [0]
        i = 0

        while i < len(data_notnull):
            #保证最后一个bin里面的num>min_num
            if (len(data_notnull) - i > min_num):
                i = i + min_num
                tmp = sorted_col[i]

                if tmp==sorted_col_set[-1]:
                    index1.append(len(sorted_col))
                    bin_point.append(tmp)
                    break

                index1.append(sorted_col.index(sorted_col_set[sorted_col_set.index(tmp)+1])-1)
                i = sorted_col.index(sorted_col_set[sorted_col_set.index(tmp) + 1]) - 1
                tmp=sorted_col[i]
                bin_point.append(tmp)
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

        print bin_point

        ################################################# calc odds ##########################################################
        index1[0] = index1[0] - 1
        group=[]
        group_label=[]
        bin_N_good=[]
        logodds = []
        for i in range(0, len(index1) - 1):
            group.append(sorted_col[index1[i] + 1: index1[i + 1] + 1])
            group_label.append(label[index1[i] + 1: index1[i + 1] + 1])
            alllength=len(sorted_col[index1[i] + 1: index1[i + 1] + 1])
            goodman=sum(label[index1[i] + 1: index1[i + 1] + 1])
            bin_N_good.append([alllength,goodman])
            if alllength==goodman:
                cur_odds = 9
            elif (goodman == 0):  # if the bin only has bad
                cur_odds = -9
            else:
                odds_origin = math.log(goodman * 1.0 / (alllength - goodman))
                cur_odds = min(max(-9, odds_origin), 9)
            logodds.append(cur_odds)
        output[col_name]=[bin_point,logodds]

        if (sum(data[col_name].isnull()) > 0): #if the column has null value
            null_N_good = [sum(data[col_name].isnull()),
                           sum(map(lambda x, y: x & y, data[col_name].isnull(), data['label_profit'] == 1))]
            if(null_N_good[0]==null_N_good[1]):#if null only has good
                null_logodds=9
            elif(null_N_good[1]==0):#if null only has bad
                null_logodds=-9
            else:
                null_logodds_origin = math.log(null_N_good[1] * 1.0 / (null_N_good[0] - null_N_good[1]))
                null_logodds=min(max(-9, null_logodds_origin), 9)
            output[col_name].append(null_logodds)

    ################################################## output pickle ##########################################################
    Path=pickle_path+"/"+SP+'_dictionary1.pkl'
    file = open(Path, 'wb')
    pickle.dump(output, file)
    file.close()
    return output

def  calc_odds(dataframe_my,bin_dict,pickle_path,SP):
    fea_list = dataframe_my.columns
    output = {}

    for k in range(0, len(fea_list) - 2):
        col_name = fea_list[k]
        print col_name

        data = dataframe_my[[col_name, 'label']]
        data_notnull = data[-data[col_name].isnull()]
        index = numpy.argsort(data_notnull[col_name])
        sorted_col = data_notnull.iloc[index, 0]

        label = data_notnull.iloc[index, 1]  # sort label in the same order as column value
        label = list(label)

        index_plot=[]
        bin_point=bin_dict[col_name]
        for values in bin_point:
            index_plot.append (len(sorted_col[sorted_col<values]))

        group = []
        group_label = []
        bin_N_good = []
        logodds = []
        for i in range(0, len(index_plot) - 1):
            group.append(sorted_col[index_plot[i] + 1: index_plot[i + 1] + 1])
            group_label.append(label[index_plot[i] + 1: index_plot[i + 1] + 1])
            alllength = len(sorted_col[index_plot[i] + 1: index_plot[i + 1] + 1])
            goodman = sum(label[index_plot[i] + 1: index_plot[i + 1] + 1])
            bin_N_good.append([alllength, goodman])
            if alllength == goodman:
                cur_odds = 9
            elif (goodman == 0):  # if the bin only has bad
                cur_odds = -9
            else:
                odds_origin = math.log(goodman * 1.0 / (alllength - goodman))
                cur_odds = min(max(-9, odds_origin), 9)
            logodds.append(cur_odds)
        output[col_name] = [bin_point, logodds]

        if (sum(data[col_name].isnull()) > 0):  # if the column has null value
            null_N_good = [sum(data[col_name].isnull()),
                           sum(map(lambda x, y: x & y, data[col_name].isnull(), data['label_profit'] == 1))]
            if (null_N_good[0] == null_N_good[1]):  # if null only has good
                null_logodds = 9
            elif (null_N_good[1] == 0):  # if null only has bad
                null_logodds = -9
            else:
                null_logodds_origin = math.log(null_N_good[1] * 1.0 / (null_N_good[0] - null_N_good[1]))
                null_logodds = min(max(-9, null_logodds_origin), 9)
            output[col_name].append(null_logodds)
    print logodds

    ################################################## output pickle ##########################################################
    Path = pickle_path + "/" + SP + '_dictionary1.pkl'
    file = open(Path, 'wb')
    pickle.dump(output, file)
    file.close()
    return output



def odds_transform_single(Dict, dataframe, SP,out_path):
    fea_list = dataframe.columns # feature list, the first one is label
    output_df=pd.DataFrame()
    output_df['label']=dataframe['label']

    for k in range(1,len(fea_list)-2):
        col_name = fea_list[k]
        print col_name
        bin_point = Dict[col_name][0]
        logodds = Dict[col_name][1]
        data=dataframe[[col_name]]
        data_notnull = data[-data[col_name].isnull()]
        col = list(data_notnull[col_name])

        pos = np.digitize(col, bin_point)#the bin number for each non-null point in the column
        bin=pd.Series([None]*len(data))
        bin[-data[col_name].isnull()]=pos
        bin[bin> len(bin_point) - 1] = len(bin_point) - 1#if  point is larger than the last bin point, assign it to the last bin
        bin[bin< 1] = 1#if  point is smaller than the first bin point, assign it to the first bin

        odds= pd.Series([None]*len(data))
        odds[-bin.isnull()]=[logodds[i - 1] for i in bin[-bin.isnull()]]#assign logodds for each non-null point

        if (sum(data[col_name].isnull()) > 0): #if has null value
            # print Dict[col_name]
            null_logodds = Dict[col_name][2]
            odds[data[col_name].isnull()] = null_logodds

        output_df[col_name] = odds

    Path = out_path + '/data_odds.pkl'
    file = open(Path, 'wb')
    pickle.dump(output_df, file)
    file.close()

    #output csv format
    txtPath = out_path +'/' +SP+'data.txt'
    output_df.to_csv(txtPath, sep='\t', index=False, header=False)

    # profitpath=out_path +'/'+SP+ 'weight.txt'

    # new=dataframe[dataframe['label_profit']==0]['label_profit']
    dataframe=pd.DataFrame(dataframe)

    # 将负值的特征
    # dataframe[dataframe['label_profit']==0] = 3
    # print dataframe['label_profit']
    # dataframe['label_profit'].to_csv(profitpath, sep='\t', index=False, header=False)
    #
    #将收益作为权重
    # pd.DataFrame(map(abs, [(dataframe['profit']*0.0001)])).T.to_csv(profitpath, sep='\t', index=False, header=False)

    #将负收益的2倍作为权重
    # print pd.DataFrame(np.array(dataframe[dataframe['profit']<0])*1.5)
    # dataframe[dataframe['profit']<0]=pd.DataFrame(np.array(dataframe[dataframe['profit']<0])*2)
    # pd.DataFrame(map(abs, [(dataframe['profit'] * 0.0001)])).T.to_csv(profitpath, sep='\t', index=False, header=False)


def load_data(path):
    return pd.read_csv(path, error_bad_lines=False)

def spit_term_dataframe(dataframe_my,term):

    start_data = '2014-09-01 00:00:00'
    end_data = '2016-12-01 00:00:00'
    spilt_data='2016-10-01 00:00:00'
    start_data = datetime.datetime.strptime(start_data, '%Y-%m-%d %H:%M:%S')
    end_data = datetime.datetime.strptime(end_data, '%Y-%m-%d %H:%M:%S')

    print len(dataframe_my)

    # dataframe_my=dataframe_my[dataframe_my['loan_term']==term]

    dataframe_my = dataframe_my[dataframe_my["inspection_time"] >= str(start_data)]
    dataframe_my = dataframe_my[dataframe_my["inspection_time"] <str(end_data)]
    print dataframe_my["inspection_time"]
    traindata=dataframe_my[dataframe_my["inspection_time"] < str(spilt_data)]
    testdata = dataframe_my[dataframe_my["inspection_time"] >= str(spilt_data)]
    print len(dataframe_my),len(traindata),len(testdata)
    return dataframe_my,traindata,testdata

def radio_final():

    dict1={'month_income': [-0.5, 1474.7049999999999, 1860.415, 2339.585, 2867.1350000000002, 3517.6999999999998, 4731.2700000000004, 8338.75, 16890.415000000001, 40466.665000000001, 4202633.8300000001],
        'QUERY_TIMES2': [-0.5, 0.5, 1.5, 2.5, 4.5, 44.5],
        'credit_grade': [139.5, 202.5, 214.5, 226.5, 234.5, 246.5, 258.5, 272.5, 288.5, 487.5],
        'card_interrupt': [-0.5, 0.5, 5.5],
        'credit_level': [0.5, 4.5, 5.5, 6.5, 8.5, 9.5, 11.5],
        'GENDER': [-0.5, 0.5, 1.5],
        'CREDIT_CARD_MAX_NUM': [-0.5, 0.5, 6.5, 13.5, 23.5, 41.5, 277.5],
        'latest_month_income': [-0.5, 0.5, 1631.3800000000001, 2393.25, 3320.5, 5056.1499999999996, 11004.5, 5000000.5],
        'LONG_REPAYMENT_TERM': [2.5, 14.0, 30.0, 5000.5],
        'gap': [-0.5, 1409.5, 2004.6199999999999, 2662.5749999999998, 3471.2200000000003, 4583.96, 8224.7350000000006, 18905.0,27329.25, 2180488.0],
        'org_type': [0.5, 2.5, 4.0, 7.5, 8.5],
        'in_city_years': [-0.5, 3.5, 5.5, 8.5, 13.5, 24.5, 30.5, 39.5, 60.5],
        'HOUSE_CONDITION': [-0.5, 0.5, 2.0, 4.5, 5.5],
        'n2': [-0.5, 1386.095, 1799.7750000000001, 2250.3400000000001, 2800.04, 3479.3800000000001, 4655.7800000000007, 8447.6599999999999, 19450.0, 50215.0, 7580000.5],
        'n3': [-0.5, 0.5, 1629.96, 2354.0, 3412.2049999999999, 7899.2250000000004, 1915244.5],
        'MAX_CREDIT_CARD_AGE': [-0.5, 0.5, 6.5, 13.5, 23.5, 41.5, 277.5],
        'age': [23.5, 27.5, 29.5, 32.5, 35.5, 39.5, 44.5, 50.5, 63.5],
        'mean': [-0.5, 792.57749999999999, 1075.2925, 1388.825, 1728.6533333299999, 2250.1949999999997, 3316.25, 6084.4166666649999, 11705.416666649999, 27329.25, 2180488.0]
    }
    dictfinal={
        'JOB_POSITION': [[0, 3.5, 4.5, 8.5, 12],
                          [-0.35326013113083127, -0.04705584240888856, 0.291581979897109, 0.6054050764643625]],
        'MAR_STATUS': [[0, 1.5, 3.5, 4.5], [-0.20833608560186492, 0.1433830171605609, 0.5615924638112952],
                        0.5826982193828119],
        'HOUSE_CONDITION': [[-0.5, 0.5, 2.0, 4.5, 5.5],
                            [-0.07016288528676613, 0.22928103524983137, 0.06101019667808261, -0.1258406339951676]],
        'HAS_INSURANCE': [[-0.5, 1.5], [0.06864219361721181], -0.09925495809534127], 'credit_level': [
            [-0.5, 0.065000000000000002, 0.095000000000000001, 0.125, 0.155, 0.185, 0.23499999999999999,
             0.35499999999999998, 594.5],
            [0.5744325068140043, 0.3667477945088356, 0.20981266984024283, 0.10584475318579127, 0.00686956603275394,
             -0.10431624196324416, -0.258256168567967, -0.3252592936957339], 0.09531017980432493],
        'card_interrupt': [[-0.5, 0.5, 2.5, 6.5], [1.3675837506746225, 0.7013358762265107, 0.6354543742176602],
                           -1.5429483749094837], 'credit_grade': [
            [-0.5, 0.065000000000000002, 0.095000000000000001, 0.125, 0.155, 0.185, 0.23499999999999999,
             0.35499999999999998, 594.5],
            [0.5744325068140043, 0.3667477945088356, 0.20981266984024283, 0.10584475318579127, 0.00686956603275394,
             -0.10431624196324416, -0.258256168567967, -0.3252592936957339], 0.09531017980432493],
        'MAX_CREDIT_CARD_AGE': [[-0.5, 0.5, 10.5, 22.5, 35.5, 50.5, 71.5, 22837.5],
                                [0.03555842332327052, -0.11753400066568041, -0.05534008682635291, -0.010277816845870282,
                                 0.07429406367406827, 0.1694841896606149, 0.26105652771157395], -0.09925495809534127],
        'age': [[8.5, 26.5, 28.5, 30.5, 33.5, 37.5, 42.5, 47.5, 63.5],
                [-0.11168849149826203, -0.01701127503601286, -0.00023595340029840313, 0.010644975656693532,
                 0.007142031556928892, -0.03691461350753689, 0.0025031302181184748, 0.44974611246808954],
                0.3470310676715481],
        'in_city_years': [[-0.5, 4.25, 7.0999999999999996, 11.5, 22.5, 27.5, 31.5, 38.5, 47.5, 99.5],
                          [-0.11847236742740658, -0.07114523076309412, -0.00263551061800834, 0.011669991245220144,
                           -0.07869884161718031, 0.059144541470565706, 0.06030638776017971, 0.050654617124726874,
                           0.591789010281887]], 'latest_month_income': [
            [-0.5, 0.0050000000000000001, 2104.0450000000001, 2650.0100000000002, 3200.0100000000002,
             3856.0699999999997, 4853.1350000000002, 7618.96, 3881383881.8000002],
            [0.7521082653557067, 0.6178030689841109, 1.3524368068923645, 1.2171933674264752, 1.3297119571550136,
             1.2079532691436508, 0.9500919850477333, 0.5640758425787299], -1.5429483749094837], 'Unnamed: 0': [
            [26588.5, 149685.5, 300249.5, 450709.5, 601814.5, 752632.5, 903082.5, 1054015.5, 1204720.5, 1355578.5,
             1478333.5],
            [0.05014692663072703, 0.06254864896593369, 0.0497555061166899, 0.05657391947557569, 0.07287132659245175,
             0.07371662254046805, 0.06846412670223272, 0.08035917369292134, 0.08772844602051708, 0.08367100266122705]],
        'org_type': [[0.5, 2.5, 4.0, 7.5, 8.5],
                     [0.5698495987082572, -0.10484271093464109, 0.23813405641681157, -0.6800118650710169],
                     -0.2719337154836418],
        'APPLY_MAX_AMOUNT': [[1.5, 30000.5, 50500.0, 70150.0, 80500.0, 101000.0, 900000000.5],
                             [0.07319010481944477, -0.004539650373724019, 0.12941263802018974, 0.10119280621633787,
                              0.05738081416259685, 0.16919806818534247]], 'n1': [
            [-0.5, 1739.0050000000001, 2256.5050000000001, 2700.0299999999997, 3147.3900000000003, 3641.0500000000002,
             4300.0300000000007, 5354.4500000000007, 8779.4500000000007, 54073592.5],
            [0.2847640662960668, 1.196231654803676, 1.3243436666809905, 1.2712927133649532, 1.301417478251409,
             1.225723620300101, 1.1596812431462924, 0.8394668603104908, 0.454763475706141], -1.5429483749094837],
        'month_income': [
            [-0.5, 2024.0149999999999, 2419.3274999999999, 2817.2687500000002, 3219.3600000000001, 3666.6800000000003,
             4268.2199999999993, 5282.79, 9236.8349999999991, 43379431.170000002],
            [0.3396512399154813, 1.4038772975138858, 1.3500982494205331, 1.3467479222132608, 1.3563798045483675,
             1.3217860327632278, 1.1947923309026294, 0.5822442267187221, 0.3740418524567015], -1.5429483749094837],
        'LOAN_TYPE': [[-0.5, 0.5, 1.5], [-0.008281440953034718, 0.5881052884510946]], 'gap': [
            [-0.5, 633.02499999999998, 1450.7249999999999, 2150.0150000000003, 2869.2950000000001, 3664.1949999999997,
             4790.6499999999996, 6958.0149999999994, 15768.73, 3881383881.8000002],
            [1.2047063095321011, 1.4565392336610705, 1.3899002238024096, 1.2969788356454106, 1.1427054180703906,
             1.1132860662652928, 0.9722478492511656, 0.38799507694054325, 0.3125020141760906], -1.5429483749094837],
        'n4': [[-0.5, 0.02, 2495.0900000000001, 3243.0550000000003, 4195.0349999999999, 64686159.5],
               [0.623792120849338, 1.1595382832986925, 1.4847475319257752, 1.4896221086448167, 1.143079102277014],
               -1.5429483749094837],
        'GENDER': [[-0.5, 0.5, 1.5], [0.2869382128983955, -0.039131276071102006], 1.2992829841302609],
        'INDUSTRY1': [[0.5, 5.5, 8.5, 9.5, 11.5, 13.5, 22.5],
                      [0.4688634122062379, 0.29856872038863, 0.45164240132627614, -0.04829690749722107,
                       0.10001786322734646, 0.014530949561700365], -7.579018918646213],
        'QUERY_TIMES2': [[-0.5, 0.5, 1.5, 2.5, 4.5, 6.5, 9.5, 80.5],
                         [0.13653079078641475, 0.13798220664966432, 0.09492151220202971, 0.058501674271881465,
                          0.031854831632383764, 0.004621118673621234, -0.07380932391078741], -0.09925495809534127]


    }


    # dict_test = {
    #     "org_type":[-0.5,1.5,2.5,4,5.5,7.5,9],
    #     "GENDER":[-0.5,0.5,1.5],
    #     "card_interrupt":[-0.5,1.5,2.5,3.5,6.5],
    #     "HOUSE_CONDITION":[-0.5,1.5,3.5,4.5,5.5],
    #     "month_income":[-0.5,],
    #     "n4", "mean", "n2",
    #             "in_city_years", \
    #             "credit_level", "MAX_CREDIT_CARD_AGE", "CREDIT_CARD_MAX_NUM", "latest_month_income",
    #             "QUERY_TIMES2", "gap", "QUERY_TIMES9", "JOB_POSITION", "MAX_LOAN_AGE", "INDUSTRY1", "LONG_REPAYMENT_TERM",
    #             "age", "MAX_CREDIT_LINE", "credit_grade",
    #     'month_income': [-0.5, 1474.7049999999999, 1860.415, 2339.585, 2867.1350000000002, 3517.6999999999998,
    #                           4731.2700000000004, 8338.75, 16890.415000000001, 40466.665000000001, 4202633.8300000001],
    #          'QUERY_TIMES2': [-0.5, 0.58, 0.78, 2.5, 4.5, 44.5],
    #          'credit_grade': [139.5, 202.5, 214.5, 226.5, 234.5, 246.5, 258.5, 272.5, 288.5, 487.5],
    #          'card_interrupt': [-0.5, 0.5, 5.5],
    #          'credit_level': [0.5, 4.5, 5.5, 6.5, 8.5, 9.5, 11.5],
    #          'GENDER': [-0.5, 0.5, 1.5],
    #          'CREDIT_CARD_MAX_NUM': [-0.5, 0.5, 6.5, 13.5, 23.5, 41.5, 277.5],
    #          'latest_month_income': [-0.5, 0.5, 1631.3800000000001, 2393.25, 3320.5, 5056.1499999999996, 11004.5,
    #                                  5000000.5],
    #          'LONG_REPAYMENT_TERM': [2.5, 14.0, 30.0, 5000.5],
    #          'gap': [-0.5, 1409.5, 2004.6199999999999, 2662.5749999999998, 3471.2200000000003, 4583.96,
    #                  8224.7350000000006, 18905.0, 27329.25, 2180488.0],
    #          'org_type': [0.5, 2.5, 4.0, 7.5, 8.5],
    #          'in_city_years': [-0.5, 3.5, 5.5, 8.5, 13.5, 24.5, 30.5, 39.5, 60.5],
    #          'HOUSE_CONDITION': [-0.5, 0.5, 2.0, 4.5, 5.5],
    #          'n2': [-0.5, 1386.095, 1799.7750000000001, 2250.3400000000001, 2800.04, 3479.3800000000001,
    #                 4655.7800000000007, 8447.6599999999999, 19450.0, 50215.0, 7580000.5],
    #          'n3': [-0.5, 0.5, 1629.96, 2354.0, 3412.2049999999999, 7899.2250000000004, 1915244.5],
    #          'MAX_CREDIT_CARD_AGE': [-0.5, 0.5, 6.5, 13.5, 23.5, 41.5, 277.5],
    #          'age': [23.5, 27.5, 29.5, 32.5, 35.5, 39.5, 44.5, 50.5, 63.5],
    #          'mean': [-0.5, 792.57749999999999, 1075.2925, 1388.825, 1728.6533333299999, 2250.1949999999997, 3316.25,
    #                   6084.4166666649999, 11705.416666649999, 27329.25, 2180488.0]
    #          }
    return dictfinal


def odds_transtram_start(data_feature_name2,target,split):
    data_name=args.data_name
    term=args.term
    logistic_regression=args.logistic_regression

    dataframe_my = load_data(data_name)
    dataframe_my=dataframe_my[data_feature_name2]
    dataframe_my, traindata, testdata = spit_term_dataframe(dataframe_my, term)

    # print len(traindata),len(testdata)
    # traindata.to_csv(str(logistic_regression) + '/' +"train")
    # testdata.to_csv(str(logistic_regression) + '/' + "test")
    # traindata = load_data(str(logistic_regression) + '/' +"train")
    # testdata = load_data(str(logistic_regression) + '/' + "test")

    # ouoputdict1 = split_bin_su(traindata, str(logistic_regression), "train" + str(term))
    # print ouoputdict1
    #
    # # pkl_file = open(str(logistic_regression)+"/train"+str(term)+"_dictionary1.pkl", 'rb')
    # # ouoputdict1=dict(pickle.load(pkl_file))
    ouoputdict1=radio_final()
    # ouoputdict1=calc_odds(traindata,bin_dict,str(logistic_regression),"train"+str(term))
    #

    odds_transform_single(ouoputdict1, traindata, "train_" , str(logistic_regression))
    odds_transform_single(ouoputdict1, testdata, "test_" , str(logistic_regression))


if __name__ == '__main__':


    data_feature_name=[ 'GENDER','score_card_type','gap','mean','grade_version','HIGHEST_DIPLOMA','no_interrupted_card_num','card_interrupt',
                         'n1','LONG_REPAYMENT_TERM', 'age','ACCEPT_MOTH_REPAY',  'in_city_years','credit_grade','INDUSTRY1',
                        'latest_month_income','repay_income_ratio','MAX_CREDIT_CARD_AGE', 'MAX_LOAN_AGE', 'month_income', 'LOAN_COUNT', 'QUERY_TIMES2',
                        'label','label_profit','profit',"issue_date","loan_term"]


    data_feature_name1=['org_type','GENDER',"card_interrupt","HOUSE_CONDITION","month_income","n3","mean","n2","in_city_years",\
    "credit_level","MAX_CREDIT_CARD_AGE","CREDIT_CARD_MAX_NUM","latest_month_income","QUERY_TIMES2","gap",\
    "age","LONG_REPAYMENT_TERM","credit_grade",'label','label_profit','profit',"issue_date","loan_term"]


    data_feature_name2= [ "card_interrupt",
                          "HOUSE_CONDITION",
                          "month_income",
                          "n4",
                          "n1",
                          "in_city_years",
                          "APPLY_MAX_AMOUNT",
                          "org_type",
                          "HAS_INSURANCE",
                          "MAX_CREDIT_CARD_AGE",

                          "LOAN_TYPE",
                          "latest_month_income",
                          "QUERY_TIMES2",
                          "gap",
                          "INDUSTRY1",
                          "age",
                          "GENDER",
                          "credit_grade",
                          "credit_level",
                          "inspection_time",
                          'label']
# "MAR_STATUS",
#                           "JOB_POSITION",
    odds_transtram_start(data_feature_name2,0,55000)
