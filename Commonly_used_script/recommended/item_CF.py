import math

def ItemSimilarity(train):
	# calculate co-rated users between items
	C = {}
	N = {}
	for u, items in train.items():
		for i in items:
			N[i] += 1
			for j in items:
				if i == j:
					continue
			C[i][j] += 1
	W = dict()
	for u, related_items in C.items():
		for v, cij in related_items.items():
			W[u][v] = cij / math.sqrt(N[i] * N[j])
	return W

def Recommendation(train, user_id, W, K):
    rank = dict()
    ru = train[user_id]
    for i,pi in ru.items():
        for j, wj in sorted(W[i].items(), key=itemgetter(1), reverse=True)[0:K]:
            if j in ru:
                continue
            rank[j] += pi * wj
    return rank

#ItemCF-IUF  对很多过于活跃的用户,为了避免相似度矩阵 过于密集,在实际计算中直接忽略他的兴趣列表
def ItemSimilarity(train):
    #calculate co-rated users between items
    C = dict()
    N = dict()
    for u, items in train.items():
         for i in items:
             N[i] += 1
             for j in items:
                 if i == j:
                     continue
                 C[i][j] += 1 / math.log(1 + len(items) * 1.0)
    #calculate finial similarity matrix W
    W = dict()
    for u,related_items in C.items():
	    for v, cij in related_items.items():
		    W[u][v] = cij / math.sqrt(N[i] * N[j])
    return W

function CalculateSimilarity(entity-items):
	w = dict()
    ni = dict()
    for e,items in entity_items.items():
        for i,wie in items.items():
            addToVec(ni, i, wie * wie)
            for j,wje in items.items():
                addToMat(w, i, j, wie, wje)
    for i, relate_items in w.items():
 relate_items = {x:y/math.sqrt(ni[i] * ni[x]) for x,y in relate_items.items()}