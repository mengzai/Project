# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import pickle
import math
import datetime
import numpy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_name',type=str,default='data_ori/dh_csxd_train_new',help='training data in csv format')
parser.add_argument('--test_name',type=str,default='data_ori/dh_csxd_test_title',help='training data in csv format')
parser.add_argument('--dh_all',type=str,default='data/dh_all',help='training data in csv format')
parser.add_argument('--logistic_regression', type=str, default='logistic_regression',
                    help='training data in csv format')
# parser.add_argument('--term',type=str,default="version",help='training data in csv format')
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


def split_bin_su(dataframe_my):
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

    fea_list = dataframe_my.columns  # feature list, the first one is label
    output = {}  # output dictionary

    for k in range(1, len(fea_list)):  # process one column each time
        col_name = fea_list[k]
        print col_name

        data = dataframe_my[[col_name, 'label']]
        data_notnull = data[-data[col_name].isnull()]
        # sorted_col = sorted(data_notnull[col_name])#sort the column value asc
        index = numpy.argsort(data_notnull[col_name])
        sorted_col = data_notnull.iloc[index, 0]
        sorted_col = list(sorted_col)
        label = data_notnull.iloc[index, 1]  # sort label in the same order as column value
        label = list(label)

        ##########################################  bin_point  #####################################################
        # set the maximum number of bins
        num_bin = 10
        # minimum number of points in each bin
        min_num = int(len(data_notnull) * 1.0 / num_bin)
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

        bin_point[0] = bin_point[0] - 0.5
        bin_point[-1] = bin_point[-1] + 0.5

        for i in range(1, len(bin_point) - 1):
            bin_point[i] = (sorted_col[index1[i]] + sorted_col[index1[i] + 1]) * 1.0 / 2

        ################################################## calc odds ##########################################################

        index1[0] = index1[0] - 1
        # group of value
        group = [sorted_col[index1[i] + 1: index1[i + 1] + 1] for i in range(0, len(index1) - 1)]
        # group of label
        group_label = [label[index1[i] + 1: index1[i + 1] + 1] for i in range(0, len(index1) - 1)]
        index1[0] = index1[0] + 1

        bin_N_good = [[len(group[i]), sum(group_label[i])] for i in range(0, len(index1) - 1)]
        logodds = []

        for i in range(0, len(index1) - 1):
            if (bin_N_good[i][1] == bin_N_good[i][0]):  # if the bin only has good
                cur_odds = 9
            elif (bin_N_good[i][1] == 0):  # if the bin only has bad
                cur_odds = -9
            else:
                odds_origin = math.log(bin_N_good[i][1] * 1.0 / (bin_N_good[i][0] - bin_N_good[i][1]))
                cur_odds = min(max(-9, odds_origin), 9)
            logodds.append(cur_odds)

        output[col_name] = [bin_point, logodds]

        if (sum(data[col_name].isnull()) > 0):  # if the column has null value
            null_N_good = [sum(data[col_name].isnull()),
                           sum(map(lambda x, y: x & y, data[col_name].isnull(), data['label'] == 1))]
            if (null_N_good[0] == null_N_good[1]):  # if null only has good
                null_logodds = 9
            elif (null_N_good[1] == 0):  # if null only has bad
                null_logodds = -9
            else:
                null_logodds_origin = math.log(null_N_good[1] * 1.0 / (null_N_good[0] - null_N_good[1]))
                null_logodds = min(max(-9, null_logodds_origin), 9)
            output[col_name].append(null_logodds)

    ################################################## output pickle ##########################################################


    Path = 'data/pickle/dictionary.pkl'
    file = open(Path, 'wb')
    pickle.dump(output, file)
    file.close()
    return output


def split_bin(dataframe_my, pickle_path, SP, num_bin=10):
    fea_list = dataframe_my.columns
    output = {}

    for k in range(0, len(fea_list) - 2):
        col_name = fea_list[k]
        print col_name
        data = dataframe_my[[col_name, 'label']]
        data_notnull = data[-data[col_name].isnull()]

        # 将数据从小到大进行排序,并将label排序   返回数据从小到大的索引值,并根据索引值得到数据从小到大的排序值同理将lable进行排序,并转化为list
        index = np.argsort(data_notnull[col_name])
        sorted_col = list(data_notnull.iloc[index, 0])
        sorted_col_set = sorted(list(set(sorted_col)))
        label = list(data_notnull.iloc[index, 1])  # sort label in the same order as column value

        ##########################################  bin_point  #####################################################
        # 根据num_bin  得到每个bin里面最少有多少个值,并将第一个bin_point为开始值
        min_num = int(len(data_notnull) * 1.0 / num_bin)
        bin_point = [sorted_col[0]]
        index1 = [0]
        i = 0

        while i < len(data_notnull):
            # 保证最后一个bin里面的num>min_num
            if (len(data_notnull) - i > min_num):
                i = i + min_num
                tmp = sorted_col[i]

                if tmp == sorted_col_set[-1]:
                    index1.append(len(sorted_col))
                    bin_point.append(tmp)
                    break

                index1.append(sorted_col.index(sorted_col_set[sorted_col_set.index(tmp) + 1]) - 1)
                i = sorted_col.index(sorted_col_set[sorted_col_set.index(tmp) + 1]) - 1
                tmp = sorted_col[i]
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

        bin_point[0] = bin_point[0] - 0.5
        bin_point[-1] = bin_point[-1] + 0.5

        for i in range(1, len(bin_point) - 1):
            bin_point[i] = (sorted_col[index1[i]] + sorted_col[index1[i] + 1]) * 1.0 / 2

        print bin_point

        ################################################# calc odds ##########################################################
        index1[0] = index1[0] - 1
        group = []
        group_label = []
        bin_N_good = []
        logodds = []
        for i in range(0, len(index1) - 1):
            group.append(sorted_col[index1[i] + 1: index1[i + 1] + 1])
            group_label.append(label[index1[i] + 1: index1[i + 1] + 1])
            alllength = len(sorted_col[index1[i] + 1: index1[i + 1] + 1])
            goodman = sum(label[index1[i] + 1: index1[i + 1] + 1])
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

    ################################################## output pickle ##########################################################
    Path = pickle_path + "/" + SP + '_dictionary1.pkl'
    file = open(Path, 'wb')
    pickle.dump(output, file)
    file.close()
    return output


def calc_ods(bin_point_dict, dataframe):
    """
        This function calc odds given binpoints.
        INPUT:
          Dict: disctionary.
            key is feature name,
            value is a list which contains bin points
          dataframe: train data. The first column must be label.
          out_path : LOCATION to store the output
        OUTPUT:
          pickle(dataframe).The first column is label, followed by features(which in the same order of input)
        """
    import pandas as pd
    import numpy as np
    import math

    fea_list = dataframe.columns  # feature list, the first one is label
    output_dict = {}
    for k in range(1, len(fea_list)):  # process one column each time
        col_name = fea_list[k]
        data = dataframe[[col_name, 'label']]
        data_notnull = data[-data[col_name].isnull()]
        index = np.argsort(list(data_notnull[col_name]))
        sorted_col = data_notnull.iloc[index, 0]
        sorted_col = list(sorted_col)
        label = data_notnull.iloc[index, 1]  # sort label in the same order as column value
        label = pd.Series(list(label))
        num_nonull = len(data_notnull)

        bin_point = bin_point_dict[col_name]
        pos = np.digitize(sorted_col, bin_point)
        bin=pd.Series(pos)
        bin[bin > len(bin_point) - 1] = len(bin_point) - 1  # if  point is larger than the last bin point, assign it to the last bin
        bin[bin < 1] = 1

        logodds=[]
        for i in range(1,len(bin_point)):
            cur_len=sum(bin==i)
            cur_good=sum(label[bin==i])

            if (cur_len==cur_good):  # if the bin only has good
                cur_odds = 9
            elif (cur_good == 0):  # if the bin only has bad
                cur_odds = -9
            else:
                odds_origin = math.log(cur_good * 1.0 / (cur_len - cur_good))
                cur_odds = min(max(-9, odds_origin), 9)
            logodds.append(cur_odds)

        output_dict[col_name]=[bin_point,logodds]

        if(len(data)-num_nonull>0):
            num_null = len(data) - num_nonull
            null_good = sum(data[data[col_name].isnull()]['label'])
            null_N_good = [num_null, null_good]
            if (null_N_good[0] == null_N_good[1]):  # if null only has good
                null_logodds = 9
            elif (null_N_good[1] == 0):  # if null only has bad
                null_logodds = -9
            else:
                null_logodds_origin = math.log(null_N_good[1] * 1.0 / (null_N_good[0] - null_N_good[1]))
                null_logodds = min(max(-9, null_logodds_origin), 9)
            output_dict[col_name].append(null_logodds)
        print col_name


    print output_dict



def odds_transform_single(Dict, Dataframe_my, SP):
    output_df = pd.DataFrame()

    output_df['label'] = Dataframe_my['label']

    # for k in range(1, len(fea_list)):
    for (key, val) in Dict.items():
        col_name = key
        print col_name
        bin_point = Dict[col_name][0]
        logodds = Dict[col_name][1]
        data = Dataframe_my[[col_name]]
        data_notnull = data[-data[col_name].isnull()]
        col = list(data_notnull[col_name])

        pos = np.digitize(col, bin_point)  # the bin number for each non-null point in the column
        bin = pd.Series([None] * len(data))
        bin[-data[col_name].isnull()] = pos
        bin[bin > len(bin_point) - 1] = len(
            bin_point) - 1  # if  point is larger than the last bin point, assign it to the last bin
        bin[bin < 1] = 1  # if  point is smaller than the first bin point, assign it to the first bin

        odds = pd.Series([None] * len(data))
        odds[-bin.isnull()] = [logodds[i - 1] for i in bin[-bin.isnull()]]  # assign logodds for each non-null point

        if (sum(data[col_name].isnull()) > 0):  # if has null value
            # print Dict[col_name]
            null_logodds = Dict[col_name][2]
            odds[data[col_name].isnull()] = null_logodds

        output_df[col_name] = odds

    # Path =  'data/odds/data_odds.pkl'
    # file = open(Path, 'wb')
    # pickle.dump(output_df, file)
    # file.close()

    # output csv format
    txtPath = 'lr/' + SP + 'data.txt'
    output_df.to_csv(txtPath, sep='\t', index=False, header=False)

    # profitpath=out_path +'/'+SP+ 'weight.txt'

    # new=dataframe[dataframe['label_profit']==0]['label_profit']
    # dataframe = pd.DataFrame(dataframe)

    # 将负值的特征
    # dataframe[dataframe['label_profit']==0] = 3
    # print dataframe['label_profit']
    # dataframe['label_profit'].to_csv(profitpath, sep='\t', index=False, header=False)
    #
    # 将收益作为权重
    # pd.DataFrame(map(abs, [(dataframe['profit']*0.0001)])).T.to_csv(profitpath, sep='\t', index=False, header=False)

    # 将负收益的2倍作为权重
    # print pd.DataFrame(np.array(dataframe[dataframe['profit']<0])*1.5)
    # dataframe[dataframe['profit']<0]=pd.DataFrame(np.array(dataframe[dataframe['profit']<0])*2)
    # pd.DataFrame(map(abs, [(dataframe['profit'] * 0.0001)])).T.to_csv(profitpath, sep='\t', index=False, header=False)

def load_data(path):
    return pd.read_csv(path, error_bad_lines=False)

def spit_term_dataframe(dataframe_my, term):
    start_data = '2014-09-01 00:00:00'
    end_data = '2016-12-01 00:00:00'
    spilt_data = '2016-10-01 00:00:00'
    start_data = datetime.datetime.strptime(start_data, '%Y-%m-%d %H:%M:%S')
    end_data = datetime.datetime.strptime(end_data, '%Y-%m-%d %H:%M:%S')

    print len(dataframe_my)

    # dataframe_my=dataframe_my[dataframe_my['loan_term']==term]

    dataframe_my = dataframe_my[dataframe_my["inspection_time"] >= str(start_data)]
    dataframe_my = dataframe_my[dataframe_my["inspection_time"] < str(end_data)]

    traindata = dataframe_my[dataframe_my["inspection_time"] < str(spilt_data)]
    testdata = dataframe_my[dataframe_my["inspection_time"] >= str(spilt_data)]
    print len(dataframe_my), len(traindata), len(testdata)
    return dataframe_my, traindata, testdata


def radio_final():
    dict = {
             'n2': [[-0.5, 1747.5250000000001, 2246.9849999999997, 2675.0150000000003, 3091.04, 3540.2150000000001,
                4130.7749999999996, 5015.125, 6987.5500000000002, 98875962.0],
               [2.67765165558229, 2.6655084420726176, 2.744844193606126, 2.707516440747931, 2.679281864658041,
                2.731001674405777, 2.7068399350519448, 2.7039972945305575, 2.5075962843098867], 2.146632525439464],
             'ORG_TYPE': [[0.5, 1.5, 2.5, 4.0, 8.5],
                          [2.9994768435492642, 2.740925617855421, 2.3374807823856667, 2.2105611741808717],
                          2.7101313673060345],
             'JOB_POSITION': [[0.5, 2.5, 3.5, 4.5, 10.5],
                              [2.4245617102385784, 2.503459114611636, 2.4636227815916447, 2.6603276851680078],
                              0.6931471805599453],
             'IS_OVERDUE': [[-0.5, 0.5, 1.5], [2.519491088059256, 2.6564296362730704], 2.3657920828759154],

             'CREDIT_CARD_NUM': [[-0.5, 0.5, 1.5, 2.5, 4.5, 6.5, 9.5, 140.5],
                                 [2.3185382960751073, 2.505084582023845, 2.52930078543552, 2.600164985969407,
                                  2.6817447130459464, 2.6884929140780147, 2.7270563671897894], 2.207274913189721],
             'FEE_MONTHS': [[-0.5, 0.5, 20150724.5], [2.4958009876100236, 2.781650048776597], 0.6931471805599453],
             'HOUSE_CONDITION': [[-0.5, 0.5, 2.0, 5.5], [2.425036794094975, 2.621104045899289, 2.482266909048653],
                                 0.6931471805599453],
             'risk_industry': [[-0.5, 0.5, 1.5], [2.7167336763124426, 2.38467383234924], 9],
             'LOAN_MAX_NUM': [[0.5, 8.5, 17.5, 27.5, 40.5, 55.5, 68.5, 86.5, 106.5, 12090.5],
                              [2.132625716129097, 2.373856171924924, 2.451871859013758, 2.472839180387883,
                               2.5477585880352525, 2.576643365910668, 2.57327454863369, 2.7216473027246177,
                               2.856162396551491], 2.5721003634982837],
             'card_cnt': [[0.5, 6.5], [2.660282757709141], 2.146632525439464],
             'HOUSING_LOAN_NUM': [[-0.5, 0.5, 16.5], [2.5532900764355824, 2.5330041193011397], 2.207274913189721],
             'GENDER': [[-0.5, 0.5, 1.5], [2.896534862839494, 2.4193003320940343]],
             'APPLY_MAX_AMOUNT': [[1.5, 31000.0, 50500.0, 70150.0, 80500.0, 102350.0, 900000000.5],
                                  [2.914974275679954, 2.4863564230380035, 2.3614753820445107, 2.2903771022212545,
                                   2.4480684740799625, 2.819533867435644]],
             'MARRIAGE': [[0.5, 1.5, 4.5], [2.4959953070748284, 2.55691051685919], 2.7759744570339118],
             'break_times': [[-0.5, 0.5, 6.5], [2.8706314903534227, 2.638395365791733], 2.146632525439464],
             'MAX_DIPLOMA': [[0.5, 1.5, 2.5, 4.5], [2.2893481822177635, 2.560009602360796, 3.0929846505006258], 9],
             'QUERY_TIMES2': [[-0.5, 0.5, 1.5, 2.5, 3.5, 5.5, 8.5, 61.5],
                              [2.772954152538019, 2.7635558834674145, 2.7447949293403644, 2.6545904341830644,
                               2.5605312970558107, 2.392009078863482, 1.9953905584350167], 2.207274913189721],
             'age': [[20.5, 25.5, 28.5, 31.5, 34.5, 38.5, 42.5, 46.5, 91.5],
                     [2.5174174561937153, 2.4424984407357324, 2.4392329306019356, 2.4363875985387593,
                      2.4579066558997624, 2.491193462516081, 2.5947704920610057, 2.8782573568339536],
                     2.5084371471981943],
             'HAS_CAR': [[-0.5, 0.5, 1.5], [2.472635963253981, 2.7785056776743575], 0.6931471805599453],
             'IN_CITY_YEARS': [[-0.5, 5.5, 10.25, 20.5, 27.5, 31.5, 36.5, 42.5, 99.5],
                               [2.224692607745542, 2.56172347280615, 2.6596147753940027, 2.545146730165357,
                                2.538948591402078, 2.4543584057652974, 2.533327475740159, 2.7800179724904055],
                               0.6931471805599453]
             }

    return dict


def odds_transtram_start(data_feature_name):
    # 抽取数据,  data_feature_name:特征名;    dataframe_my:dataframe:特征的数据
    data_name = args.data_name
    test_name = args.test_name
    data_name_df = load_data(data_name)
    test_name_df = load_data(test_name)

    test_name_df=test_name_df[-test_name_df['GENDER'].isnull()]
    test_name_df.to_csv('lr/test')
    test = load_data('lr/test')
    dataframe_my, traindata, testdata = spit_term_dataframe(dataframe_my)

    ouoputdict1 = split_bin_su(dataframe_my)
    print ouoputdict1
    # #
    # pkl_file = open('data/pickle/dictionary.pkl', 'rb')
    # ouoputdict1=dict(pickle.load(pkl_file))
    # print ouoputdict1
    ouoput=radio_final()
    print ouoput

    odds_transform_single(ouoput, data_name_df, "train_" )
    odds_transform_single(ouoput, test, "test_" )
def find_data(data_feature_name):
    data_name = args.data_name
    test_name = args.test_name
    data_name_df = load_data(data_name)
    test_name_df = load_data(test_name)

    listab = [data_name_df, test_name_df]
    dh_all=pd.concat(listab)



    # data_name_df = data_name_df[data_feature_name]
    print len(dh_all)
    data_name_df.to_csv('data/all',index=False)
    #
    # test_name_df = test_name_df[data_feature_name]
    # print len(test_name_df)
    # test_name_df.to_csv('data/test',index=False)



def ues():
    data_name=args.test_name
    data=load_data(data_name)

    clom=['label', 'APPLY_MAX_AMOUNT',  'JOB_POSITION', 'ACCEPT_MOTH_REPAY', 'SCORE',
     'FEE_MONTHS','LONG_REPAYMENT_TERM','GENDER',
     'apply_times', 'age', 'RECRUITMENT_DATE', 'IN_CITY_YEARS', 'pledge_num', 'n1', 'n5', 'end_balance', 'break_times',
     'MAXOVER_DAYS',
     'MAXOVER_AMOUNT', 'loan_purpose', 'mobile_type', 'ORG_TYPE', 'MAX_DIPLOMA', 'MARRIAGE', 'HAS_CAR',
     'HOUSE_CONDITION','INDUSTRY1','risk_industry','IS_OVERDUE','MAX_TOTAL_OVERDUE_DAYS','default_times','CREDIT_CARD_NUM',
'NOMAL_CARD_NUM','CLOSED_CARD_NUM','NON_ACTIVATE_CARD_NUM','CREDIT_CARD_MAX_NUM','LOAN_MAX_NUM','MAX_CREDIT_CARD_AGE',
'MAX_CREDIT_LINE', 'MAX_LOAN_AGE','TOTAL_OVERDUE_NUM_L','QUERY_TIMES2','OVERDUE_CARD_NUM','MAX_OVERDUE_NUM_C','HOUSING_LOAN_NUM']

    datanew=data[clom]
    datanew.fillna(-999999, inplace = True)
    datanew.to_csv('testhhhhh.txt',index=False,dialect='excel')
    #
    data_sub = data[['label', 'MAR_STATUS', 'JOB_POSITION']]
    dict_instruct = {'MAR_STATUS': [0, 1.5, 3.5, 4.5], 'JOB_POSITION': [0, 3.5, 4.5, 8.5, 12]}
    calc_ods(dict_instruct, data_sub)
    #
    # col_name = 'age'
    # data1 = pd.read_csv('dh_csxd_train')
    # data_notnull1 = data1[-data1[col_name].isnull()]
    # data2 = pd.read_csv('dh_csxd_test')
    # data_notnull2 = data2[-data2[col_name].isnull()]
    # tmp1 = list(data_notnull1[col_name])
    # tmp2 = list(data_notnull2[col_name])
    #
    # tmp1.extend(tmp2)
    # # #print len(data)-len(data_notnul
    # # print sorted(list(data_notnull1[col_name]))[int(len(data_notnull1)*0.85)]
    # print sorted(tmp1)[int(len(tmp1) * 0.85)]





if __name__ == '__main__':
    ues()
    # data_feature_name = ['label','APPLY_MAX_AMOUNT', 'ACCEPT_MOTH_REPAY', 'a_ANNUAL_INCOME', 'n2', 'n4','TOTAL_CARD_OVERDUE_AMOUNT',
    #     'LOAN_COUNT', 'FEE_MONTHS', 'apply_times', 'GENDER', 'age',
    #     'mobile_type', 'ORG_TYPE', 'RECRUITMENT_DATE'
    #     , 'JOB_POSITION', 'MAX_DIPLOMA', 'MARRIAGE', 'HAS_CAR', 'HOUSE_CONDITION', 'IN_CITY_YEARS',
    #     'INDUSTRY1', 'contact_num', 'pledge_num',
    #     'card_cnt',  'break_times', 'risk_industry', 'CREDIT_CARD_NUM', 'NON_ACTIVATE_CARD_NUM',
    #     'CREDIT_CARD_MAX_NUM', 'LOAN_MAX_NUM', 'MAX_CREDIT_CARD_AGE',
    #     'MAX_CREDIT_LINE', 'TOTAL_OVERDUE_NUM_L','QUERY_TIMES2', 'MAX_OVERDUE_NUM_C', 'HOUSING_LOAN_NUM', 'IS_OVERDUE']
    #
    # data_feature_name = ['label','APPLY_MAX_AMOUNT', 'ACCEPT_MOTH_REPAY', 'a_ANNUAL_INCOME', 'n2', 'n4','TOTAL_CARD_OVERDUE_AMOUNT']
    #
    # feature=['label','TOTAL_OVERDUE_NUM_L','TOTAL_CARD_OVERDUE_AMOUNT','SCORE','risk_industry','RECRUITMENT_DATE','QUERY_TIMES2','pledge_num','OVERDUE_CARD_NUM',
    #          'ORG_TYPE','NON_ACTIVATE_CARD_NUM','NOMAL_CARD_NUM','n5','n1','MONTH_PAYMENT','mobile_type','MAXOVER_DAYS','MAXOVER_AMOUNT','MAX_TOTAL_OVERDUE_DAYS',
    #          'MAX_OVERDUE_NUM_C','MAX_DIPLOMA','MAX_CREDIT_LINE','MAX_CREDIT_CARD_AGE','MARRIAGE','LONG_REPAYMENT_TERM','loan_purpose','LOAN_MAX_NUM','LOAN_COUNT',
    #          'JOB_POSITION','IS_OVERDUE','IS_CITY_SAME','INDUSTRY1','IN_CITY_YEARS','HOUSING_LOAN_NUM','HOUSE_CONDITION','HAS_CAR','GENDER','FEE_MONTHS','end_balance'
    #          ,'default_times','CREDIT_CARD_NUM','CREDIT_CARD_MAX_NUM','contact_num','CLOSED_CARD_NUM','break_times','apply_times','APPLY_MAX_AMOUNT','age','ACCEPT_MOTH_REPAY']
    #
    # odds_transtram_start(data_feature_name)
    # # find_data(feature)


