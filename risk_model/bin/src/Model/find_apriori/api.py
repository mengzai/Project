#-*- coding: UTF-8 -*-
import math
import csv
import json
import Apriori

#创建初始列表；
#输入，【【月息通，月满盈，月】，【月，月满盈】，【】】：即包含集合复集合
#输出【frozenset([月满盈])】；：即冰冻的Set字典项
def credateC1(DtaaSet):
    C1=[]
    for transcation in DtaaSet:
        for item in transcation:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset,C1)

#将C1变为L1
#输入：数据集Ck，最小支持度，包含候选集合的列表
#输出：满足最小支持度的新集合；每个候选集的所占比例
def scanD(Dataset,Ck,minSupport):
    sscnt={}
    #遍历数据项和遍历刚开始的列表
    for tid  in Dataset:
        for can in Ck:
            #判断数据项是否在列表中
            if can.issubset(tid):
                #通过字典计算每个列表的候选项所占比例
                if not sscnt.has_key(can):
                    sscnt[can]=1
                else:sscnt[can]+=1
    numItems=float(len(Dataset))
    retList=[]
    supportData={}
    print "mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm"
    for key in sscnt:
        support=sscnt[key]/numItems
        ##通过字典计算每个列表的候选项所占比例删除不符合支持度的候选项
        if support>=minSupport:
            retList.insert(0,key)
        supportData[key]=support
    return retList,supportData

#L1为一级剔除不满足最小支持度的一级频繁项

#输入：Lk 第k级频繁项，项集个数k    Lk：【1：0.5】带有比例的
#输出：Lk为将频繁项中的每个项与其它项重新组合得到二级频繁项得到Ck   Ck不带比例的
def apriorGen(Lk,k):#create Ck
    retList=[]
    lenLk=len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1=list(Lk[i])[:k-2]
            L2=list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1==L2:
                #python的∪集合
                retList.append(Lk[i]|Lk[j])
    return retList



def apriori(dataSet,minSupport):
    #奖励C1初始化的频繁项
    C1=credateC1(dataSet)
    #将原始数据变为frozenset的类型
    D=map(frozenset,dataSet)
    print 2
    #删掉不读和最小支持度的频繁项并得到每个频繁项的比例
    L1,supportData=scanD(D,C1,minSupport)
    L=[L1]
    k=2
    #循环
    finash = []
    while(len(L[k-2])>0):
        #循环得到Ck，Lk
        Ck=apriorGen(L[k-2],k)
        Lk,supk=scanD(D,Ck,minSupport)
        print k
        print Lk
        print supk
        supportData.update(supk)
        L.append(Lk)
        finash.append(Lk)
        k+=1
    return finash,supportData



#输入：L:频繁项集列表，supportData：频繁项中每项  minconf:最小可信度阀值
#输出：

def getraterules(L,supportData,minconf=0.7):
    biglist=[]   #输出最后的每个规则集
    for i in range(1,len(L)):
        for freqSet in L[i]:                                                                 #从二级频繁项开始索引
            H1=[frozenset([item]) for item in freqSet]                                       #将索频繁项分开变为frozenset类型
            if i>1:
                rulesFromConseq(freqSet,H1,supportData,biglist,minconf)                       #当为d大于二级级频繁项时：找到频繁项里面的规则
            else:
                calcConf(freqSet,H1,supportData,biglist,minconf)                            # i=1 #当为二级频繁项时：直接找到二级频繁项里面的规则
    return biglist



#输入：freseq：为二级频繁项合集   ,H：为二级频繁项分开之后的集合,supportData频繁项每个项的支持度比例,br1：规则的集合,minconf：这个规则的置信度
#输出：计算置信度并返回单个的频繁项
#目的：计算股则的置信度
def calcConf(freseq,H,supportData,br1,minconf=0.7):
    prunedH=[]
    for conseq in H:
        conf=supportData[freseq]/supportData[freseq-conseq]    #计算置信度：即为两个的支持度比例
        if conf>=minconf:                                     #如果置信度>最小置信度
            print freseq-conseq,'-->',conseq,'conf:',conf
            br1.append((freseq-conseq,conseq,conf))            #得到余下/目标的置信度
            prunedH.append(conseq)
    return prunedH                                            #返回逐个的频繁项



#输入：freseq：为二级频繁项合集   ,H：为二级频繁项分开之后的集合,supportData频繁项每个项的支持度比例,br1：规则的集合,minconf：这个规则的置信度
# 二级频繁项中找规则
#直接调用calcConf得到对应的置信度

def rulesFromConseq(freseq,H,supportData,br1,minconf=0.7):
    m=len(H[0])
    if (len(freseq)>(m+1)):
        hmp1=apriorGen(H,m+1)
        hmp1=calcConf(freseq,hmp1,supportData,br1,minconf)
        if (len(hmp1)>1):
            rulesFromConseq(freseq,hmp1,supportData,br1,minconf)

def loaddata():
    count=0
    f=open("data_1_sp",'rb')
    for line in f.readlines():
        count+=1
    print count

    product_list=[]
    file = open("data_1_sp", 'rb')
    for row_data1 in file.readlines():
        row_data1=row_data1.strip()
        row_data1=row_data1.replace("\t"," ")
        row_data1= row_data1.split(" ")
        producer_dic = []
        for i in range(len(row_data1)):
            if  row_data1[i] == '万'or row_data1[i] == '通过' or row_data1[i].isdigit() or row_data1[i] == '期' or row_data1[i] == '同意' or row_data1[i] == ',' or row_data1[i] == '1' or row_data1[i] == '-1' or row_data1[i] == '，' or row_data1[i] == '；'or row_data1[i] == '.' or row_data1[i] == ';' or row_data1[i] == '。':
                pass
            else:
                producer_dic.append(row_data1[i])
        product_list.append(producer_dic)
    return product_list
if __name__ == '__main__':
    product_list=loaddata()
    Dataset = product_list
    l,supp= apriori(Dataset,minSupport=0.01)


    product_list0=Apriori.loaddata()
    Dataset0=product_list0
    l0, supp0 = Apriori.apriori(Dataset0, minSupport=0.01)

    l0_dict=[]
    for i in list(l0):
        for j in i:
            l0_dict.append(list(j))
            # print json.dumps(list(j), encoding='utf-8', ensure_ascii=False), supp[j]

    for m in list(l):
        for n in m:
            if list(n) not in l0_dict:
                print json.dumps(list(n), encoding='utf-8', ensure_ascii=False), supp[n]


