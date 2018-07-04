#-*- coding: UTF-8 -*-
import math
import sys

type = sys.getfilesystemencoding()

#评分预测
#input records:user,item,rui u对i的实际评分,pui  u对i的算法预测出来的评分
#RESE
def  RESE(records):
	return math.sqrt(sum([(rui-pui)*(rui-pui) for u,i,rui,pui in records]))/float(len(records))
#MSE
def MSE(records):
	return sum([(rui - pui)  for u, i, rui, pui in records]) / float(len(records))

#TOP_N:召回率 准确率
#input  test:表示在测试集上做预测 test 为字典值user,item ,N表示top N 前N个推荐
def  PrecisionREcall(test,N):
	hit=0
	n_recall=0
	n_precision=0
	for user ,items in test.items():
		rank = Recommend(user, N) # rank 表示针对该user 进行推荐的前N个item
		hit += len(rank & items)  # 推荐准确的item
		n_recall += len(items)
		n_precision += N
	return [hit / (1.0 * n_recall), hit / (1.0 * n_precision)]

#覆盖率 :信息熵 及基尼系数可以用来评测覆盖率的指标
def GiniIndex(p):
	j=1
	n = len(p)
	G=0
	for item, weight in sorted(p.items(), key=itemgetter(1)):
		G += (2 * j - n - 1) * weight
	return G / float(n - 1)

def Coverage(train, test, N):
    recommend_items = set()
    all_items = set()
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)
        rank = GetRecommendation(user, N)
        for item, pui in rank:
            recommend_items.add(item)
    return len(recommend_items) / (len(all_items) * 1.0)

#新颖度
def Popularity(train, test, N):
         item_popularity = dict()
         for user, items in train.items():
              for item in items.keys():
                  if item not in item_popularity:
                      item_popularity[item] = 0
                  item_popularity[item] += 1
         ret = 0
         n=0
         for user in train.keys():
              rank = GetRecommendation(user, N)
              for item, pui in rank:
                  ret += math.log(1 + item_popularity[item])
                  n += 1
         ret /= n * 1.0
		return ret