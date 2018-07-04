import random
import math
#分割数据
#input data 总数据集
def  Spit_data(data,M,k,seed):
	train=[]
	test=[]
	random.seed(seed)
	for (user ,item) in data:
		if random.randint(0,M)==k:
			test.append([user,item])
		else:
			train.append([user,item])
	return train,test


#找到u v属于倒排表中 k个物品 对应的用户列表  将user->items 更改为item->users
#即扫描每个用户 对应的用户列表,将用户列表中的两两用户的对应的C[u][v] +1  最后可以得到所有用户之间不为0的C[u][v]
#input:train (user,item)
def UserSimilarity(train):
	item_user={}
	for u ,items in train.items():
		for i in items.keys():
			if i not in item_user:
				item_user[i]=set()
			item_user[i].add(u)
	#以上我们得到啦 item->users 的字典集合,下面针对一个item 一旦有u v重合进行+1
	C=dict()
	N=dict()
	for i ,users in item_user.item():
		for u in  users:
			N[u]+=1 #计算每个user 所喜欢的items 数据
			for v in users:
				if u==v:
					continue
				C[u][v]+=1 # 得到uv之间的公共喜欢的产品数目
	#计算相似度
	W={}
	for u,relates_users in C.items():
		for v, cuv in relates_users.items():
			W[u][v]=cuv/math.sqrt(N[u]*N[v])
	return W

#通过计算相似度的结果 可以得到 UserCF 推荐算法
#input user 针对用户user  根据W相似度 可以得到 选取top k  ,rvi代表用户v对物品i的兴趣,为使用的是单一行为的隐反馈数 据,所以所有的rvi=1
def Recommend(user, train, W,K):
	rank = dict()
	interacted_items = train[user]
	for v, wuv in sorted(W[user].items, key=itemgetter(1), reverse=True)[0:K]:
		for i, rvi in train[v].items:
			if i in interacted_items:
				# we should filter items user interacted before
				continue
			rank[i] += wuv * rvi
	return rank

