#!/usr/bin/env python
# -*- coding:utf-8 -*- 

import numpy as np
import pandas as pd
import xgboost as xgb
# import plot_roc_ks
import datetime
import pickle

model_features = ['label', 'phone', 'call_times', 'connect_times', 'has_callin', 'has_staff_hangup', 'avg_waittime',
                  'min_waittime', 'max_waittime', 'avg_onlinetime', 'min_onlinetime', 'max_onlinetime', 'province',
                   'callresult','str_zhengxin', 'str_jujie', 'str_zhuce', 'str_mingtian', 'str_mendian', 'str_kaolv',
                  'str_feilv', 'str_daka', 'str_guanji', 'emotion', 'weekday', 'avg_comments_cnt', 'onlinetime_gap',
                  'online_ascending_num', 'online_decsending_num', 'waittime_ascending_num', 'waittime_decsending_num',
                  'month_nums_in', 'beta_online', 'beta_wait', 'sex', 'level','loanamount', 'age',
                  ##new feature,'has_car','house',
                  'repayterm_ratio', 'overdue_all_day', 'breach_amortisation', 'repayamount_ratio',
                  'degree', 'marriage', 'PINCOME', 'HIGH_CREDIT', 'CREDIT_LEVEL', 'credit_card_num', 'has_children',
                  'issocial', 'call_mean_10', 'call_std_10', 'jietong_ratio_10', 'wait_mean_10',
                  'wait_std_10',  'is_jinjian',
                  'lasttonow_call_daynums','lasttonow_jietong_daynums',
                  'callresult_mean_10', 'callresult_mean_5', 'callreuslt_previous','cr_conti',
                  'duration_midu_10', 'jietong_midu_20','duration_midu_20',
                  'jietong_midu_30', 'duration_midu_30','callreuslt_beta','call_mean_5', 'call_std_5',  'jietong_ratio_5',
                  'wait_mean_5','wait_std_5','call_mean_20', 'call_std_20', 'call_midu_20', 'jietong_ratio_20',
                  'wait_mean_20','wait_std_20','call_mean_30', 'call_std_30', 'call_midu_30', 'jietong_ratio_30',
                  'wait_mean_30','wait_std_30','trans_age', 'trans_province', 'age_province', 'age_sex', 'sex_province',
                  'city_low100','intamortisation']
                  ##delete  ,'call_midu_10', 'city_low50','call_midu_5','jietong_midu_5','jietong_midu_10',
                  ##


def output_final(data, Path):
    f = open(Path, "w")
    data.to_csv(Path, index=False, header=True)
    f.close()


def load_data(filename):
    data = pd.read_csv(filename, sep='\t', error_bad_lines=False).fillna(value=-999999)
    return data


def dateRange(start, end, step=1, format="%Y-%m-%d"):
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    days = (strptime(end, format) - strptime(start, format)).days
    return [strftime(strptime(start, format) + datetime.timedelta(i), format) for i in xrange(0, days, step)]


def calc_weight_new(id, label, w):
    id_dict = {}
    for item in id:
        if (not id_dict.has_key(item)):
            id_dict[item] = 1
        else:
            id_dict[item] = id_dict[item] + 1
    weight = map(lambda x: 1.0 / id_dict[id[x]] if label[x] == 0 else  w * 1.0 / id_dict[id[x]], range(len(id)))
    return weight


def get_date_list(month):
    date_list = []
    if month == 2:
        date_list = dateRange("2017-02-06", "2017-03-01")
    elif month == 3:
        date_list = dateRange("2017-03-01", "2017-04-03")
    elif month == 4:
        date_list = dateRange("2017-04-03", "2017-05-02")
    elif month == 5:
        date_list = dateRange("2017-05-02", "2017-06-01")
    elif month == 6:
        date_list = dateRange("2017-06-01", "2017-07-01")
    return date_list


def main():
    month = 6  # 表示测试集选取月份
    date_list = get_date_list(month)

    train = load_data('./dianxiao_train')  ##读入训练集数据
    train = train[model_features]  ## 只选入部分特征
    print train.columns
    test = load_data('./dianxiao_test' )
    test = test[model_features]  ## 只选入部分特征

    print 'calc weight'
    ###计算训练集权重
    weight_train = calc_weight_new(train['phone'], train['label'], 25)  # 最大42

    ######
    col_names = train.columns  ##特征名称列表
    print col_names[2:]
    ######

    train_label, train_feat = train.values[0:, 0], train.values[0:, 2:]
    dtrain = xgb.DMatrix(train_feat, label=train_label, feature_names=col_names[2:], weight=weight_train)

    test_label, test_feat = test.values[0:, 0], test.values[0:, 2:]  ##
    dtest = xgb.DMatrix(test_feat, label=test_label, feature_names=col_names[2:])

    param = {}
    param['booster'] = 'dart'
    param['max_depth'] = 4  # 2-5
    param['learning_rate'] = 0.14  # 0.05-1
    param['objective'] = 'binary:logistic'  ######目标函数,可以自己写一个函数
    # param['subsample'] = 0.9
    # param['colsample_bytree'] = 0.6  # 0.9
    param['nthread'] = 8
    param['silent'] = True
    param['eval_metric'] = 'auc'
    # param['rate_drop'] = 0.1
    # param['skip_drop'] = 0.5
    param['min_child_weight'] = 120  # 120
    num_round = 800  # 150

    print('Training')
    clf = xgb.train(param, dtrain, num_round, evals=[(dtrain,'train'),(dtest, "test")], early_stopping_rounds=50)
    # return
    ####保存模型
    with open('./model_v2.pkl', 'wb') as file:
        pickle.dump(clf, file)
    file.close()

    importances = clf.get_score()
    ###打印出特征重要性
    print sorted(importances.iteritems(), key=lambda d: d[1], reverse=True)

    ####################训练集AUC结果###############################################
    # with open('model_v2.pkl', 'rb') as file:
    #     model_v2 = pickle.load(file)
    print 'train auc'
    dtrain = xgb.DMatrix(train_feat, label=train_label, feature_names=col_names[2:])
    predict_test = clf.predict(dtrain)
    plot_obj_test = plot_roc_ks.Plot_KS_ROC(list(train['label']), list(predict_test), './plot/', 'train')  # population
    plot_obj_test.plot_ks()
    plot_obj_test.plot_roc()

    return
    ####################################################  使用early stopping ##############################################
    # 自定义loss function :
    # f(x, y) = f loss objective where x = prediction, y = label
    # gradient => derivative of f by x
    # hessian => derivative of gradient by x
    # def fair_obj(preds, dtrain):
    #     fair_constant=100
    #     labels = dtrain.get_label()
    #     x = (preds - labels)
    #     den = abs(x) + fair_constant
    #     grad = fair_constant * x / (den)
    #     hess = fair_constant * fair_constant / (den * den)
    #     return grad, hess

    # watchlist = [(dtrain, 'train'), (dtest, 'val')]#利用测试集去early stopping
    # model = xgb.train(params, dtrain, num_boost_round=100000, evals=watchlist, early_stopping_rounds=10, obj=fair_obj)



    ###################################################### 测试集结果 #####################################################

    auc_list = []  ##用来保存每天预测的AUC
    ######    对每天的数据进行预测
    for current_date in date_list:
        ##############    读入当天的测试集
        cur_test = load_data(
            './final/final_%s/%s' % (month, current_date))
        cur_test = cur_test[model_features]  ## 只选入部分特征
        test_label, test_feat = cur_test.values[0:, 0], cur_test.values[0:, 2:]  ##
        dtest = xgb.DMatrix(test_feat, label=test_label, feature_names=col_names[2:])

        print('Testing')

        ###########预测概率
        predict_test = clf.predict(dtest)

        print('Plotting')
        ################  画出KS与ROC图,计算KS与AUC
        plot_obj_test = plot_roc_ks.Plot_KS_ROC(list(cur_test['label']), list(predict_test), './plot/',
                                                str(current_date))  # population
        plot_obj_test.plot_ks()
        cur_auc = plot_obj_test.plot_roc()
        auc_list.append(cur_auc)

        ###############  输出预测结果
        # out = pd.DataFrame()
        # out['phone'] = cur_test['phone']
        # out['label'] = cur_test['label']
        # # out['is_fk'] = cur_test['fk_label']
        # out['predict'] = predict_test
        # Path = './prob/' + str(current_date)
        # f = open(Path, "w")
        # out.to_csv(Path, sep='\t', index=False, header=True)
        # f.close()
        print current_date

    output_final(pd.Series(auc_list), 'auc_%s_v2.csv' % month)  # auc输出


if __name__ == '__main__':
    main()
