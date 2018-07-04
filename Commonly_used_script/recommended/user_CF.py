#-*- coding: UTF-8 -*-
import math
import csv
import  operator
import random
import gettext
import StringIO
import json
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pylab as pl
import numpy as np
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from collections import defaultdict
import copy
import codecs
import sys
import re
import time
import cx_Oracle
from threading import Thread
from threading import Lock
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
#import  cx_Oracle
type = sys.getfilesystemencoding()

"""
=========================================
Feature importances with forests of trees
=========================================

This examples shows the use of forests of trees to evaluate the importance of
features on an artificial classification task. The red bars are the feature
importances of the forest, along with their inter-trees variability.

As expected, the plot suggests that 3 features are informative, while the
remaining are not.
"""
#读取数据并且将空格的地方填写值

def feature_importances(dataset):
    X=dataset.iloc[:,0:8]
    y=dataset['admit']
    ###########  预测特征+结果显示
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)
    z=forest.fit(X, y)

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking


    feature_list=[]
    for f in range(8):
        #print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        feature_list.append([indices[f],importances[indices[f]]])
        feature_list.sort(reverse=True)
    print("Feature ranking:")
    print(feature_list)
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(8), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(8), indices)
    plt.xlim([-1, 8])
    plt.show()

    return  feature_list

#########################################
#随机森林分类   输出随机森林分类的结果和方式
def RandomForestClassifie(dataset):
    print("#############随机森林分类###################")
    df = pd.DataFrame(dataset, columns=['FHIGHEST_EDUCATION','﻿"FSEX"','CLIENTLEVELE1','FINVEST_STATUS','ZIP1','RESOURE','FTRADE','FRANK','FANNUAL'])
    df['species'] = dataset['admit']

    ########## 数据集的划分  随机划分
    df['is_train'] = np.random.uniform(0, 1, len(data)) <= .75
    train, test = df[df['is_train']==True], df[df['is_train']==False]

    features = df.columns[:8]
    clf = RandomForestClassifier(n_jobs=2)
    y, _ = pd.factorize(train['species'])
    clf.fit(train[features], y)

    for j in df.columns[3:4]:
                featureSet=set(df["%s"%j].drop_duplicates())
    preds = [clf.predict(test[features])]
    c=pd.crosstab(test['species'], preds, rownames=['actual'], colnames=['preds'])
    print(c)
    return c





#格式化成字典数据
    #将数据分类
    #    1.用户字典test_dic：dic[用户id]=[(产品id,行为评分)...]
    #    2.产片字典 test_item_to_user：dic产品id]=[用户id1,用户id2...]
def createUserRankDic(test_rates):
    #print("###############将数据转变为两个list############################")
    user_rate_dic={
}
    item_to_user={
}
    for i in test_rates:  #i是数组的意思
        user_rank=(i[1],i[2])
        if i[0] in user_rate_dic:
            user_rate_dic[i[0]].append(user_rank) #用户打分字典
        else:
            user_rate_dic[i[0]]=[user_rank]
        if i[1] in item_to_user:
            item_to_user[i[1]].append(i[0]) #产片字典
        else:
            item_to_user[i[1]]=[i[0]]
    #print( user_rate_dic)
    #print(item_to_user)
    return user_rate_dic,item_to_user


def calcCosDistSpe(user1,user2):
    u1_u2=0.0
    for key1 in user1:
        for key2 in user2:
                u1_u2=key2[1]*key1[1]
    return u1_u2


#   相似余弦距离
#
#
#
def calcSimlaryCosDist(user1,user2):
    sum_x=0.0
    sum_y=0.0
    sum_xy=0.0
    avg_x=0.0
    avg_y=0.0
    ########key :找到userid对应的[产品和行为评分]
    #######      求得user1和user2的   d的行为评分的平均值
    """
    for key in user1:
        avg_x+=int(key[1])
    avg_x=avg_x/len(user1)

    for key in user2:
        avg_y+=int(key[1])
    avg_y=avg_y/len(user2)
    """

    for key1 in user1:
        for key2 in user2:
            if int(key1[0])==int(key2[0]) :   #key[1]代表分数，key[0]代表产品名称
                sum_xy+=(round(float(key1[1]))*(round(float(key2[1]))))
                sum_y+=(round(float(key2[1]))*(round(float(key2[1]))))
        sum_x+=(round(float(key1[1]))*(round(float(key1[1]))))
    if sum_xy == 0.0 :
        return 0
    sx_sy=math.sqrt(sum_x*sum_y)
    if sx_sy==0:
        return 0
    print(sum_xy/sx_sy)
    return sum_xy/sx_sy




########################################
#输入:之前的三个月的训练集，  和对应每个用户id
      #中间通过循环遍历每一列，
#输出:每个特征的【推荐产品，评分】，推荐产品种类
      #文件数据格式化成二维数组 List[[用户id,产品id,产品行为评分]...
def find(rates,useronly,userid1):
    list1,key1=recommendByUserFC(rates,useronly,userid1)
    return  list1,key1

#
#   计算与指定用户最相近的邻居
#   输入:指定用户ID，输入用户数据，输入物品数据
#   输出:与指定用户最相邻的邻居列表
#
def calcNearestNeighbor(userid,useronly,users_dic,item_dic):
    neighbors=useronly
    neighbors.append(userid)
    ####neighbors为给出的user对应买的物品    物品对应的用户的集合
    neighbors_dist=[]
    ####按行读取neighbors ,判断user与之的相似性
    for neighbor in neighbors:
        dist=calcCosDistSpe(users_dic[userid],users_dic[neighbor])  #calcSimlaryCosDist  calcCosDist calcCosDistSpe
        if dist>0:
            neighbors_dist.append([dist,neighbor])
    neighbors_dist.sort(reverse=True)   #输出最相邻的邻居列表
    return  neighbors_dist

def createUserRankDic(rates):

    #print("###############将数据转变为两个list############################")
    user_rate_dic={
}
    item_to_user={
}
    for i in rates:  #i是数组的意思
        #print(rates)
        user_rank=(i[1],i[2])
        if i[0] in user_rate_dic:
            user_rate_dic[i[0]].append(user_rank) #用户打分字典
        else:
            user_rate_dic[i[0]]=[user_rank]

        if i[1] in item_to_user:
            item_to_user[i[1]].append(i[0]) #产片字典
        else:
            item_to_user[i[1]]=[i[0]]
    #print( user_rate_dic)
    #print(item_to_user)
    return user_rate_dic,item_to_user

########################################
#输入:csv(id+admit+特征)，目标用户 ，k为最邻近的十个临近用户
def recommendByUserFC(rates,userid,k=10):

    #格式化成字典数据
    #将数据分类
    #    1.用户字典test_dic：dic[用户id]=[(产品id,行为评分)...]
    #    2.产片字典 test_item_to_user：dic产品id]=[用户id1,用户id2...]
    test_dic,test_item_to_user=createUserRankDic(rates)
    #寻找邻居
    #寻找到邻居后，如果邻居看过的产片不在recommend_dic列表里，则添加，并认为该邻居的评分赋予给该产片，若多个邻居看过，
    # 则该产片的评分是多位邻居的评分之和，为排除目标用户已经看过的产片
    neighbors=calcNearestNeighbor(userid[9],useronly,test_dic,test_item_to_user)
    #user_product = [ i[0] for i in test_dic[userid]] #i[0]对应了目标用户买过的产品
    recommend_dic={}
    for neighbor in neighbors:
        neighbor_user_id=neighbor[1]        #neighbors包含([dist,neighbor])列表，所以neighbor[1]表示的是用户id
        product=test_dic[neighbor_user_id]  #返回的是[产品，评分]对
        for movie in product:
            if movie[0] not in recommend_dic: #movie[0]返回的是所有的产片
               #print(recommend_dic[movie[0]])
               recommend_dic[movie[0]]=movie[1]
               #print(recommend_dic[movie[0]])
                #neighbor[0]表示相似度
            else:
                recommend_dic[movie[0]]=int(recommend_dic[movie[0]])+int(movie[1])  #产片：评分1+评分2+.。。。
                #print(recommend_dic[movie[0]])
    #print len(recommend_dic)
    #建立推荐列表
    recommend_list=[]
    key1=[]
    #print(recommend_dic)
    for key in recommend_dic:  #key是产品名字
        #print key
        if key not in key1:
            key1.append(key)
        recommend_list.append([recommend_dic[key],key])
    recommend_list.sort(reverse=True)
    #print("#################################返回推荐产品列表[评分+产品]")
    return  recommend_list,key1
    #print("##################################输出目标用户的喜欢的产片")
    #print(user_product)
    #print("################################# 目标用户产片列表  即目标用户对应买的产品 则这个产品都有谁买")
    #print(test_item_to_user)
    #print("#################################  目标用户的邻居列表[相似度+用户]")
    #print(neighbors)
    #return recommend_list,user_product,test_item_to_user,neighbors #返回推荐产片列表，目标用户产片列表，产片字典，和邻居列表


#################################################333
######################################
#输入：[此产品的数量，产品的种类]
#输出：【次产品的选择概率，产品的种类】
#################################################333
######################################
#输入：[此产品的数量，产品的种类]
#输出：【次产品的选择概率，产品的种类】
def probability(score_item_list):
    print(score_item_list)
    probabilit=[]
    sum1=0
    for j in score_item_list:
        sum1+=int(j[0])
    for i in score_item_list:
        if isinstance(i[0],int):
            probabilit.append([(i[0]*1.0/sum1),i[1]])
        else:
            pass
            #print( "Error")
    print(probabilit)


################################################
#i为目标用户的基本信息，train_dic 训练集的基本数据
#找到与之属性值最相似的客户   根据属性
#*************************************
#根据之前判断的得到每个属性的重量性基本值
def  newcustomer (train_dic,i,shuxing):
     max_score=0
     scorer=[]
     item=[]
     user=[]
     for j in train_dic:
         length=len(j)
         score=0
         for m in range(length-2):
             if  j[m]==i[m] and j[10]!='0':
                 score=score+feature_list1[m]*10
         scorer.append(score)
         #print(scorer)
         if max_score<score:
             max_score=score
     #print(scorer)
     num=0
     for m in train_dic:
         if  scorer[num]>(max_score-2) :
            if m[9] not in user:
                item.append(m[10])
                user.append(m[9])
         num+=1
     user.sort(reverse=True)
     clearest=user[:5]
     return (clearest),(item)


def  newcustomer1 (train_dic,i,shuxing):
     max_score=0
     scorer=[]
     item=[]
     user=[]
     for j in train_dic:
         length=len(j)
         score=0
         for m in range(length-2):
             if  j[m]==i[m] and j[10]!='0':
                 score=score+feature_list1[m]*10
         scorer.append(score)
         #print(scorer)
         if max_score<score:
             max_score=score
     numm=0
     for m in train_dic:
         if  scorer[numm]>(max_score-2) :
            if m[9] not in user:
                item.append(m[10])
                user.append(m[9])
         numm+=1
     user.sort(reverse=True)
     clearest=user[:5]

     num=0
     rates=[]
     useronly=[]
     for m in train_dic:
         if  scorer[num]>(max_score-shuxing) :
            if m[9] not in user:
                user.append(m[9])
                rates.append([m[9],m[10],int(scorer[num])])
                useronly.append(m[9])
         num+=1
     return clearest,rates,useronly
def probabiliter(item,dicte,user,hot_product,printstring,cleastuser):
    set1=list(set(item))
    product_set={}
    probab=[]
    sum1=0
    for j in item:
        sum1+=1

    for sm in set1:
        set_number=0
        for i in item:
                if i==sm:
                   set_number+=1
        sm1=int(sm)
        product_set[dicte[sm1]]=set_number*1.0/sum1
    new=dicte[int(user[10])]
    if user[10]==0:
        pass
    else:
        product_set[new]=0.3
    if product_set.has_key("-10"):
        if  product_set.has_key(hot_product.decode("utf_8").encode("gbk")):
             product_set[hot_product.decode("utf-8").encode("gbk")]=product_set.pop("-10")+product_set.pop(hot_product.decode("utf_8").encode("gbk"))
        else:
             product_set[hot_product.decode("utf-8").encode("gbk")]=product_set.pop("-10")
    product_set = sorted(product_set.iteritems(), key=operator.itemgetter(1), reverse=True)
    product_set=product_set[:4]
    print json.dumps(product_set,encoding='gbk', ensure_ascii=False)
    #####################将产品改为数字############3
    final.writelines(user[9]+','+json.dumps(product_set,encoding='utf-8', ensure_ascii=False)+','+printstring.decode("utf-8").encode("gbk")+','+json.dumps(cleastuser,encoding='utf-8', ensure_ascii=False)+'\n')
    return product_set


def probability(score_item_list,dicte,user,hot_product,printstring,clearest):
    product_set={}
    sum1=0
    for j in score_item_list:
        sum1+=int(j[0])
    for i in score_item_list:
        product_set[dicte[i[1]]]=i[0]*1.0/sum1
    new=dicte[int(user[10])]
    if user[10]==0:
        pass
    else:
        product_set[new]=0.3
    if product_set.has_key("-10"):
        if  product_set.has_key(hot_product.decode("utf_8").encode("gbk")):
             product_set[hot_product.decode("utf-8").encode("gbk")]=product_set.pop("-10")+product_set.pop(hot_product.decode("utf_8").encode("gbk"))
        else:
            product_set[hot_product.decode("utf-8").encode("gbk")]=product_set.pop("-10")
    product_set = sorted(product_set.iteritems(), key=operator.itemgetter(1), reverse=True)
    product_set=product_set[:4]
    print json.dumps(product_set,encoding='gbk', ensure_ascii=False)
    #####################将产品改为数字############3
    final.writelines(user[9]+','+json.dumps(product_set,encoding='utf-8', ensure_ascii=False)+','+printstring.decode("utf-8").encode("gbk")+','+json.dumps(clearest,encoding='utf-8', ensure_ascii=False)+'\n')
    return product_set
#读取数据库；循环对数据库进行处理；循环抽取三个月的数据

def SQl(time1,time2):
    sql =     (" select  t.fhighest_education,       t.Fsex ,        te.client_level   clientlevele,         t.fzip_code         zip,           t.FCLIENT_SOURCE      resoure,  t.Ftrade,   t.FRANK        frank,  tc.FANNUAL_INCOME       fannual,   t.Fposition, t.lenderid,ts.product_name from tcr_cri_indi_client  t  "
           "left join tcr_cri_invest_details ts on t.lenderid = ts.lender_id "
           "inner join tcr_cri_indi_client_ecifid te on te.clientid = t.fid "
           "left join TCR_CRI_CLIENT_OTHERS tc on tc.fclient_id = t.fid "
           "where t.faccount_status = 1 "
           "and  to_char(t.FOPEN_DATE, 'yyyy-mm-dd') >=  %s"
           "and  to_char(t.FOPEN_DATE, 'yyyy-mm-dd') <=  %s"
           " order by t.lenderid"  %(time1,time2))
    return sql
def SQl1(target_user):
    sql1 =     (" select  t.fhighest_education,       t.Fsex ,        te.client_level   clientlevele,         t.fzip_code         zip,           t.FCLIENT_SOURCE      resoure,  t.Ftrade,   t.FRANK        frank,  tc.FANNUAL_INCOME       fannual,   t.Fposition, t.lenderid,ts.product_name from tcr_cri_indi_client  t  "
           "left join tcr_cri_invest_details ts on t.lenderid = ts.lender_id "
           "inner join tcr_cri_indi_client_ecifid te on te.clientid = t.fid "
           "left join TCR_CRI_CLIENT_OTHERS tc on tc.fclient_id = t.fid "
           "where t.lenderid = %s "  %target_user)
    return sql1

def SELECTDB(time1,time2):
    username = "CRM"
    pwd = "VUFzVwBS"
    dsn=cx_Oracle.makedsn('192.168.101.120','1521','cedb')
    db1=cx_Oracle.connect(username,pwd,dsn)
    outputFile = open("E://final_job//data.csv",'w') # 'wb'
    output = csv.writer(outputFile, dialect='excel')
    sql=SQl(time1,time2)
    curs2 = db1.cursor()
    curs2.execute(sql)
    cols = []

    for col in curs2.description:
        cols.append(col[0])
    output.writerow(cols)
    producer_dic=[]
    train_dic=[]
    train_user=[]
    for row_data1 in curs2: # add table rows
        row_data1=list(row_data1)
        for i in range(len(row_data1)):
            if row_data1[i]==None:
                row_data1[i]='-10'
        producer_dic.append(row_data1[10])
    productor=list(set(producer_dic))

    adict={}
    for m in range(len(productor)):
        adict[m]=productor[m]
    curs2.close
    curs3=db1.cursor()
    curs3.execute(sql)
    for row_data in curs3: # add table rows
        row_data=list(row_data)
        for i in range(len(row_data)):
            if row_data[i]==None:
                row_data[i]='-10'
        if  row_data[10]=='"PRODUCT_NAME"':
            pass
        elif row_data[10]=='""':
            row_data[10]=row_data[11].replace('""','0')
        else:
            for i in range(len(productor)):
                if row_data[10]==productor[i]:
                    row_data[10]=i
        if row_data[2]=='M2':
            row_data[2]='1'
        elif row_data[2]=='M1':
            row_data[2]='2'
        elif row_data[2]=='V':
            row_data[2]='3'
        elif row_data[2]=='SV':
            row_data[2]='4'
        elif row_data[2]=='P':
            row_data[2]='5'
        elif row_data[2]=='"CLIENTLEVELE"':
            pass
        else:
             row_data[2]='0'
        if row_data[3][1:3]=="10" or row_data[3][1:3]=="20" or row_data[3][1:3]=="50" or row_data[3][1:3]=="30":
            row_data[3]='1'
        elif row_data[3][1:4]=="404" or row_data[3][1:4]=="310" or row_data[3][1:4]=="266" or row_data[3][1:4]=="116" or row_data[3][1:4]=="315":
            row_data[3]='2'
        elif row_data[3] =='"ZIP"':
            pass
        else:
            row_data[3]='3'
        train_dic.append(row_data)
        train_user.append(row_data[9])
    outputFile.close()
    curs3.close
    db1.close()
    return train_dic,train_user,adict

def SELECTDB1(target_user):
    username = "CRM"
    pwd = "VUFzVwBS"
    dsn=cx_Oracle.makedsn('192.168.101.120','1521','cedb')
    db1=cx_Oracle.connect(username,pwd,dsn)
    outputFile = open("E://final_job//data.csv",'w') # 'wb'
    output = csv.writer(outputFile, dialect='excel')
    sql=SQl(target_user)
    curs2 = db1.cursor()
    curs2.execute(sql)
    cols = []

    for col in curs2.description:
        cols.append(col[0])
    producer_dic=[]
    train_dic=[]
    train_user=[]
    for row_data1 in curs2: # add table rows
        row_data1=list(row_data1)
        for i in range(len(row_data1)):
            if row_data1[i]==None:
                row_data1[i]='-10'
        producer_dic.append(row_data1[10])
    productor=list(set(producer_dic))

    adict={}
    for m in range(len(productor)):
        adict[m]=productor[m]
    curs2.close
    curs3=db1.cursor()
    curs3.execute(sql)
    for row_data in curs3: # add table rows
        row_data=list(row_data)
        for i in range(len(row_data)):
            if row_data[i]==None:
                row_data[i]='-10'
        if  row_data[10]=='"PRODUCT_NAME"':
            pass
        elif row_data[10]=='""':
            row_data[10]=row_data[11].replace('""','0')
        else:
            for i in range(len(productor)):
                if row_data[10]==productor[i]:
                    row_data[10]=i
        if row_data[2]=='M2':
            row_data[2]='1'
        elif row_data[2]=='M1':
            row_data[2]='2'
        elif row_data[2]=='V':
            row_data[2]='3'
        elif row_data[2]=='SV':
            row_data[2]='4'
        elif row_data[2]=='P':
            row_data[2]='5'
        elif row_data[2]=='"CLIENTLEVELE"':
            pass
        else:
             row_data[2]='0'
        if row_data[3][1:3]=="10" or row_data[3][1:3]=="20" or row_data[3][1:3]=="50" or row_data[3][1:3]=="30":
            row_data[3]='1'
        elif row_data[3][1:4]=="404" or row_data[3][1:4]=="310" or row_data[3][1:4]=="266" or row_data[3][1:4]=="116" or row_data[3][1:4]=="315":
            row_data[3]='2'
        elif row_data[3] =='"ZIP"':
            pass
        else:
            row_data[3]='3'
        train_dic.append(row_data)
        train_user.append(row_data[9])
    outputFile.close()
    curs3.close
    db1.close()
    return train_dic,train_user,adict
if __name__ == '__main__':
    feature_list1={ 0: 0.034815181644966303, 1: 0.013350767800746063,2: 0.12172884055515497,3:0.039908648970485752,4: 0.19218298545372273,5:0.31879048711918129,6: 0.14613837633582757,7: 0.13308471211991546,8:0.1}

    #输入：
    hot_product='月息通'
    target_user=15905
    final=open("E://ease-job//work//job//final1.csv",'w')       ##为刚开始输入的数据
    user_before=[]
    train_product={}
    user_before_dic=[]

    train_dic,train_user,train_prod=SELECTDB("'2015-09-01'","'2015-12-01'")
    test_dic=SELECTDB1(target_user)
    train_product=dict(train_product, **train_prod)
    user_before=train_user
    user_before_dic=user_before_dic+train_dic
    if target_user  not in user_before:
       ##################相当于新用户   这个新用户可能已经投资（  方法1）  未投资（ 新用户的冷启动）
       if test_dic[10]==0:
           print("           #####新用户 未投资  完全基于属性的")
           printstring="该用户为新用户，通过与之属性最相似的老用户曾经最喜爱的产品和曾经购买过的产品进行推荐， 最相似的用户为："
           if test_dic[9]!='LENDERID':
                    cleastuser,item=newcustomer(user_before_dic,i,shuxing=2)
                    probabiliter(item,train_product,i,hot_product,printstring,cleastuser)
       else:
           print ("#############新用户   已投资  ;基于属性和行为###########################")
           printstring="该用户为新用户并且已经投资过产品， 通过与之具有相同购买能力并且投资过包含该产品的多种产品进行推荐 最相似的用户为："
           ###################已经投资证明此人对该产品已经非常感兴趣没有其他原因则对该产品的概率直接变为0.8
           if test_dic[9]!='LENDERID':
               #在训练集中含有此数据
               if  test_dic[9]  in  train_user:
                   shuxing=7
                   cleastuser,rates,useronly=newcustomer1(user_before_dic,test_dic,shuxing)
                   list1,key1=find(rates,useronly,test_dic)
                   probability(list1,train_product,test_dic,hot_product,printstring,cleastuser)
               else:
                    print("")
                    cleastuser,item=newcustomer(user_before_dic,test_dic,shuxing=2)
                    probabiliter(item,train_product,test_dic,hot_product,printstring,cleastuser)
    else:       #当测试集的用户已经之前对其预测过了则对此用户进行替换和
       if test_dic[10]==0:
           print("           #####老用户 未投资  完全基于属性的")   #但是需要进行对此数据进行更新

           if test_dic[9]!='LENDERID':
                    cleastuser,item=newcustomer(user_before_dic,i,shuxing=2)
                    printstring="该用户为老用户并且已经投资过产品，则找到与之属性最相似的老用户曾经最喜爱的产品和曾经购买过的产品， 最相似的用户为："
                    probabiliter(item,train_product,i,hot_product,printstring,cleastuser)
           else:
               pass
       else:
           print ("#############老用户   已投资  ;基于属性和行为###########################")
           ###################已经投资证明此人对该产品已经非常感兴趣没有其他原因则对该产品的概率直接变为0.3
           if test_dic[9]!='LENDERID':
               shuxing=8
               clearest,rates,useronly=newcustomer1(user_before_dic,test_dic,shuxing)
               list1,key1=find(rates,useronly,test_dic)
               printstring="该用户为老用户并且已经投资过产品， 通过与之具有相同购买能力并且投资过包含该产品的多种产品进行推荐 最相似的用户为："
               probability(list1,train_product,test_dic,hot_product,printstring,clearest)