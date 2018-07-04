#-*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import pickle
import math
import datetime
import  numpy
import argparse
import matplotlib.pyplot as plt
from compiler.ast import flatten
import scipy.stats as stats
import datetime
from Plot_good_radio import Plot_goodradio
parser = argparse.ArgumentParser()
parser.add_argument('--data_name',type=str,default='data_all/xxd_good_and_m7',help='training data in csv format')
parser.add_argument('--logistic_regression',type=str,default='logistic_regression',help='training data in csv format')
parser.add_argument('--term',type=str,default=24,help='training data in csv format')
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
    print fea_list
    for k in range(0, len(fea_list) - 6):#process one column each time
        col_name = fea_list[k]
        # print col_name

        data = dataframe_my[[col_name, 'label_profit']]
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


    # Path = pickle_path + "/" + SP + '_dictionary1.pkl'
    # file = open(Path, 'wb')
    # pickle.dump(output, file)
    # file.close()
    return output



def split_bin(dataframe_my, pickle_path,SP,num_bin = 10):
    fea_list = dataframe_my.columns
    output={}

    for k in range(0, len(fea_list)-6):
        col_name = fea_list[k]
        print col_name
        data = dataframe_my[[col_name, 'label_profit']]
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

def  calc_odds(dataframe_my,bin_dict,pickle_path,SP,fea_list):

    output = {}

    for k in range(0, len(fea_list)):
        col_name = fea_list[k]
        print col_name

        data = dataframe_my[[col_name, 'label_profit']]
        data_notnull = data[-data[col_name].isnull()]
        index = numpy.argsort(data_notnull[col_name])
        sorted_col = data_notnull.iloc[index, 0]

        label = data_notnull.iloc[index, 1]  # sort label in the same order as column value
        label = list(label)

        index_plot=[]
        bin_point=bin_dict[col_name]
        for values in bin_point[1]:
            print values
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
    print output

    ################################################## output pickle ##########################################################
    Path = pickle_path + "/" + SP + '_dictionary1.pkl'
    file = open(Path, 'wb')
    pickle.dump(output, file)
    file.close()
    return output



def odds_transform_single(Dict, dataframe, SP,out_path):
    fea_list = dataframe.columns # feature list, the first one is label
    output_df=pd.DataFrame()
    output_df['label_profit']=dataframe['label_profit']

    for k in range(1,len(fea_list)-6):
        col_name = fea_list[k]
        # print col_name
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

    profitpath=out_path +'/'+SP+ 'weight.txt'

    # new=dataframe[dataframe['label_profit']==0]['label_profit']
    dataframe=pd.DataFrame(dataframe)

    # 将负值的特征
    # dataframe[dataframe['label_profit']==0] = 3
    # print dataframe['label_profit']
    # dataframe['label_profit'].to_csv(profitpath, sep='\t', index=False, header=False)
    #
    #将收益作为权重
    pd.DataFrame(map(abs, [(dataframe['profit']*0.0001)])).T.to_csv(profitpath, sep='\t', index=False, header=False)

    #将负收益的2倍作为权重
    # print pd.DataFrame(np.array(dataframe[dataframe['profit']<0])*1.5)
    # dataframe[dataframe['profit']<0]=pd.DataFrame(np.array(dataframe[dataframe['profit']<0])*2)
    # pd.DataFrame(map(abs, [(dataframe['profit'] * 0.0001)])).T.to_csv(profitpath, sep='\t', index=False, header=False)


def load_data(path):
    return pd.read_csv(path, error_bad_lines=False)

def spit_term_dataframe(dataframe_my,term):

    start_data = '2013-05-01 00:00:00'
    end_data = '2014-10-01 00:00:00'
    spilt_data='2014-05-01 00:00:00'
    start_data = datetime.datetime.strptime(start_data, '%Y-%m-%d %H:%M:%S')
    end_data = datetime.datetime.strptime(end_data, '%Y-%m-%d %H:%M:%S')

    print len(dataframe_my)

    dataframe_my=dataframe_my[dataframe_my['loan_term']==term]

    dataframe_my = dataframe_my[dataframe_my["issue_date"] >= str(start_data)]
    dataframe_my = dataframe_my[dataframe_my["issue_date"] <str(end_data)]

    traindata=dataframe_my[dataframe_my["issue_date"] < str(spilt_data)]
    testdata = dataframe_my[dataframe_my["issue_date"] >= str(spilt_data)]
    print len(dataframe_my),len(traindata),len(testdata)
    return dataframe_my,traindata,testdata

def radio_final():
    GENDER_0_dict={'credit_level': [[0.5, 3.5, 4.5, 5.5, 6.5, 10.5], [1.7508398056567314, 1.768529924119193, 1.794619792169763, 1.6885271364148866, 1.595025337932073], 1.466337068793427],
                   'latest_month_income': [[-0.5, 0.029999999999999999, 1993.72, 2877.0250000000001, 4969.4300000000003, 12802.5, 10788300.5], [1.967005066324396, 2.0933507017550195, 1.963246227976464, 1.7912181972256656, 1.3069776062913032, 1.2688873488497554], 2.0120002608843035], 'MAR_STATUS.1': [[0.5, 1.5, 2.5, 4.5], [2.017629375513633, 1.6843454724241524, 1.675613947893666], 1.55814461804655], 'HOUSE_CONDITION': [[-0.5, 0.5, 2.0, 4.5, 5.5], [1.6225084952106028, 1.8993935364767054, 1.8099407985273004, 1.32090802305998], 1.6808782369135469], 'MAX_LOAN_AGE': [[-0.5, 0.5, 17.5, 226.5], [1.8261928075337965, 1.1999051366309235, 1.6672064094956982], 2.7300291078209855], 'MAX_CREDIT_CARD_AGE': [[-0.5, 0.5, 7.5, 16.5, 28.5, 46.5, 66.5, 288.5], [1.8344156637456308, 1.2322652692312586, 1.3787995307554308, 1.6433483767260357, 1.8577290444788057, 1.9532766648720759, 2.0953790037858377], 2.7300291078209855], 'card_interrupt': [[-0.5, 6.5], [1.7275493347501032]], 'APPLY_MAX_AMOUNT': [[9999.5, 32500.0, 51500.0, 71500.0, 500000000.5], [2.3295448314452685, 1.7130426540139874, 1.6956045418926535, 1.5272081137788296], 9], 'MAR_STATUS': [[0.5, 1.5, 2.5, 4.5], [2.017629375513633, 1.6843454724241524, 1.675613947893666], 1.55814461804655], 'CREDIT_CARD_MAX_NUM': [[-0.5, 0.5, 7.5, 16.5, 28.5, 46.5, 66.5, 288.5], [1.8355311268614412, 1.232548916942705, 1.3787995307554308, 1.642313983047045, 1.8530864055186889, 1.9543246083790347, 2.0953512158201946], 2.7300291078209855], 'gap': [[-0.5, 1810.0050000000001, 2339.6900000000001, 2908.1500000000001, 3629.915, 4922.9650000000001, 10401.0, 26584.5, 55707.305, 33913552.348399997], [2.304102927841481, 2.194417378861378, 2.0987280399323, 2.0545907688095073, 2.084625586467424, 1.8562448426948104, 1.3759790834080576, 1.3281022886828362, 1.2704441363385945]], 'QUERY_TIMES9': [[-0.5, 0.5, 1.5, 3.5, 5.5, 9.5, 90.5], [2.0621836250363375, 2.158360653729546, 1.896891648316788, 1.7134950580657122, 1.4946408449562787, 1.0138791876842783], 1.6250954222525062], 'credit_grade': [[-22.5, 245.5, 263.5, 277.5, 285.5, 295.5, 304.5, 316.5, 329.5, 586.5], [1.5242686933882694, 1.6402858555401367, 1.7347803793373002, 1.6219467621258092, 1.9270690119994571, 1.8866335391257731, 1.6834067928566014, 1.8565105528066435, 1.7099809665550836], 1.466337068793427], 'month_income': [[-0.5, 1667.01, 2058.835, 2476.9250000000002, 2991.7750000000001, 3796.165, 6705.5, 15019.334999999999, 30300.834999999999, 69957.169999999998, 11306644.49], [2.3451668115009934, 2.2023912124408893, 2.0490032877933846, 2.0467108787025396, 1.997272789165994, 1.9453799263493599, 1.3278648642439812, 1.3436372570592918, 1.2822387866283216, 1.2754281038211732], 2.0120002608843035], 'QUERY_TIMES2': [[-0.5, 0.5, 1.5, 2.5, 4.5, 57.5], [2.031606874411213, 2.0292632574101006, 1.7732584546689707, 1.6349886125515705, 1.075042732877349], 2.7300291078209855], 'Unnamed: 0': [[-0.5, 4349.5, 8698.5, 13047.5, 17396.5, 21745.5, 26094.5, 30443.5, 34792.5, 39141.5, 43491.5], [1.6177367152487956, 1.571563885674398, 1.6341992275767987, 1.8284533534848664, 1.7625564383339278, 1.8735278028690612, 1.8695488788443424, 1.9304950822193168, 1.6157976739500783, 1.627756492957785]], 'age': [[23.5, 27.5, 29.5, 32.5, 35.5, 39.5, 44.5, 50.5, 55.5, 63.5], [1.8254845166844436, 1.7828274205625674, 1.612329051999372, 1.5788786098589724, 1.5514283223052543, 1.5559125497519946, 1.5490470538166106, 2.004574171945005, 2.214271655397819], 1.2674331582431617], 'org_type': [[0.5, 2.5, 4.0, 7.5, 8.5], [2.209246419820173, 1.5778492231740324, 1.939477966074838, 1.378491036885131], 1.713507272328307], 'in_city_years': [[-0.5, 4.5, 7.5, 10.5, 20.5, 27.5, 32.5, 40.5, 50.5, 81.5], [1.3015082503686142, 1.5765052860193565, 1.6731276571732827, 1.7329881518126415, 1.901032321705797, 1.7716742274318704, 1.7188735709327472, 1.8236387755229686, 2.2214452925061363], 1.6808782369135469], 'n1': [[-0.5, 1580.2249999999999, 1984.7849999999999, 2409.335, 2951.6800000000003, 3782.5, 6046.7849999999999, 14309.5, 32124.5, 33913552.348399997], [2.2314539150856993, 2.225909725335078, 1.9884812323504462, 2.1076793328959464, 2.065426259062715, 1.8433215446950357, 1.4551648260946628, 1.2772074637212505, 1.2876875119996056], 2.0120002608843035], 'n3': [[-0.5, 0.25, 1812.3049999999998, 2544.085, 3838.9349999999999, 8639990.5], [1.564929612760778, 2.266125995167247, 2.108199624270847, 2.09572769098209, 1.6106513996247025], 2.0120002608843035], 'mean': [[-0.5, 844.16750000000002, 1045.9383333300002, 1262.6750000000002, 1529.3975, 1956.655, 3633.7725, 8283.5591666649998, 16645.766666650001, 5653322.4947300004], [2.3294237646041176, 2.1867932399042402, 2.0645350718837876, 2.0729907387657875, 2.0432223315303113, 1.9139718249213729, 1.35360706846183, 1.3225569761818994, 1.2768210312040025]]}
    return GENDER_0_dict


def plot_ratio(data,Label,fig_save_path):

    data=data[-data.isnull()]
    print len(data)
    fig = plt.figure(figsize=(10, 10))
    plt.hist( list(data), label=Label, color="b",alpha=0.6,rwidth=0.8)
    plt.legend()
    plt.show()
    fig.savefig(str(fig_save_path) +"_" + str(Label) + '.png', dpi=180)
def desfen(data,colom, ind):
    data_notnull = data[-data[colom].isnull()][colom]
    print len(data_notnull)
    g_dist = sorted(data_notnull)
    lenth = len(g_dist)
    info = stats.describe(data_notnull)
    print g_dist[int(ind * lenth)]

def find_data_plot_radio(feature_name,target,split):
    data_name=args.data_name



    dataframe_my = load_data(data_name)
    # dataframe_my = data[data_feature_name2]
    dataframe_my, traindata, testdata = spit_term_dataframe(dataframe_my, term)

    #将数据分为子集
    # if target==0:
    #     traindata=traindata[traindata[feature_name]<split]
    #     testdata = testdata[testdata[feature_name] < split]
    # else:
    #     traindata = traindata[traindata[feature_name] >=split]
    #     testdata = testdata[testdata[feature_name] >=split]


    print len(traindata),len(testdata)
    # Plot_goodradio(traindata, term, feature_name + str(target))
    # 画出odds化图

    return traindata,testdata



if __name__ == '__main__':
    logistic_regression = args.logistic_regression
    term = args.term

    data_feature_name=[ 'GENDER','score_card_type','gap','mean','grade_version','HIGHEST_DIPLOMA','no_interrupted_card_num','card_interrupt',
                         'n1','LONG_REPAYMENT_TERM', 'age','ACCEPT_MOTH_REPAY',  'in_city_years','credit_grade','INDUSTRY1',
                        'latest_month_income','repay_income_ratio','MAX_CREDIT_CARD_AGE', 'MAX_LOAN_AGE', 'month_income', 'LOAN_COUNT', 'QUERY_TIMES2',
                        'label','label_profit','profit',"issue_date","loan_term"]

    data_feature_name1=['org_type','GENDER',"card_interrupt","HOUSE_CONDITION","month_income","n3","mean","n2","in_city_years",\
    "credit_level","MAX_CREDIT_CARD_AGE","CREDIT_CARD_MAX_NUM","latest_month_income","QUERY_TIMES2","gap",\
    "age","LONG_REPAYMENT_TERM","credit_grade",'label','label_profit','profit',"issue_date","loan_term"]


    Gender0_feature= [

        "MAR_STATUS",
        "card_interrupt",
        'org_type',
        "age",
        "gap",
         'APPLY_MAX_AMOUNT',
         "HOUSE_CONDITION",
         "month_income",
         "n3",
         "mean",
         "n1",
         "MAR_STATUS",
         "in_city_years",
         "credit_level",
         "MAX_CREDIT_CARD_AGE",
         "CREDIT_CARD_MAX_NUM",
         "latest_month_income",
         "QUERY_TIMES2",
         "QUERY_TIMES9",
         "MAX_LOAN_AGE",
          "GENDER",
         "credit_grade",
         'label',
         "GENDER",
         'label_profit',
         'profit',
         "issue_date",
         "loan_term"]

    # 得到train,test集合
    feature_name="GENDER"
    target=1
    traindata,testdata=find_data_plot_radio("GENDER",target,0.5)
    # #
    # # traindata.to_csv(str(logistic_regression) + '/' +feature_name+"_train_"+ str(term)+"_"+str(target))
    # # testdata.to_csv(str(logistic_regression) + '/' + feature_name+"_test_"+str(term) +"_"+str(target))
    # #
    # traindata = load_data(str(logistic_regression) + '/' +feature_name+"_train_"+ str(term) +"_"+str(target))
    # testdata = load_data(str(logistic_regression) + '/' + feature_name+"_test_"+str(term) + "_"+str(target))




    #分析特征
    # colom = "LONG_REPAYMENT_TERM"
    # plot_ratio(traindata[traindata[colom] < 15][colom], colom, "hist/")
    # desfen(traindata,colom, 0.6)


    #
    traindata = traindata[Gender0_feature]
    testdata = testdata[Gender0_feature]
    traindata.to_csv(str(logistic_regression) + '/' +feature_name+"_train_"+ str(term)+"_"+str(target))
    testdata.to_csv(str(logistic_regression) + '/' + feature_name+"_test_"+str(term) +"_"+str(target))

    traindata = load_data(str(logistic_regression) + '/' +feature_name+"_train_"+ str(term) +"_"+str(target))
    testdata = load_data(str(logistic_regression) + '/' + feature_name+"_test_"+str(term) + "_"+str(target))

    ouoputdict1 = split_bin_su(traindata, str(logistic_regression), "train" + str(term))
    # print ouoputdict1

    # 找到其
    # pkl_file = open(str(logistic_regression)+"/train"+str(term)+"_dictionary1.pkl", 'rb')
    # ouoputdict1=dict(pickle.load(pkl_file))
    # ouoputdict1=radio_final()
    # fea_list=["MAR_STATUS","org_type","card_interrupt"]
    # ouoputdict1=calc_odds(traindata,ouoputdict1,str(logistic_regression),"train"+str(term),fea_list)
    #
    # ouoputdict1=radio_final()
    # print traindata
    # print ouoputdict1
    odds_transform_single(ouoputdict1, traindata, feature_name+"_train_" + str(term)+"_"+str(target), str(logistic_regression))
    odds_transform_single(ouoputdict1, testdata, feature_name+"_test_" + str(term)+"_"+str(target), str(logistic_regression))

