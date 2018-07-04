# encoding=utf8
import sys
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
def xfunc(df, datasplits=[{'name' : 'age','type':'numberic', 'values':[0,30,60]} \
                          , {'name' : 'sex','type':'discrete', 'values':[0,1]}], \
         cols=['app_hobby', 'kyc_hobby'], bins=10):
    '''df：要分析的数据
       datasplits：划分规则。type有数值型和离散型。数值型需要首尾。
       cols：需要画图的列
       bins 条形图分成的bins个数
    '''

    tmpdf = df
    plt.figure()
    figcount = 0
    k = datasplits[0]
    title = [k['name']]
    for vi in range(len(k['values'])):
        if k['type'] == 'numberic' and vi == len(k['values']) -1:
            continue
        if k['type'] == 'numberic':
            tmpdf = df[df[k['name']] > k['values'][vi]]
            tmpdf = tmpdf[tmpdf[k['name']] <= k['values'][vi+1]]
            title.append(k['values'][vi])
            title.append(k['values'][vi+1])
        else:
            tmpdf = df[df[k['name']] == k['values'][vi]]
            title.append(k['values'][vi])
        if len(datasplits) > 1:
            kk = datasplits[1]
            kktitle = [kk['name']]
            for vi in range(len(kk['values'])):
                if kk['type'] == 'numberic' and vi == len(kk['values']) -1:
                    continue
                if kk['type'] == 'numberic':
                    tmpdf2 = tmpdf[tmpdf[kk['name']] > kk['values'][vi]]
                    tmpdf = tmpdf2[tmpdf2[kk['name']] <= kk['values'][vi+1]]
                    kktitle.append(kk['values'][vi])
                    kktitle.append(kk['values'][vi+1])
                else:
                    tmpdf2 = tmpdf[tmpdf[kk['name']] == kk['values'][vi]]
                    kktitle.append(kk['values'][vi])
                legend = "_".join(map(str,title + kktitle))
                kktitle = [kk['name']]
                for col in cols:
                    n,bins,patches=plt.hist(tmpdf2[col], bins, normed=1, alpha=0.75)
                    plt.title(col + " _" + legend)
                    plt.savefig("./" + col + " _" + legend + ".png")
        else:
            legend = "_".join(map(str,title))
            n,bins,patches=plt.hist(tmpdf[col], bins, normed=1, alpha=0.75)
            plt.title(col + " _" + legend)
            plt.savefig("./" + col + " _" + legend + ".png")
        title =  [k['name']]
    # return tmpdf2 if len(tmpdf2) > 0 else None

tdf = pd.read_csv('abcd.csv', sep=',')
datasplits=[{'name' : 'age','type':'numberic', 'values':[0,30,60]} \
                          , {'name' : 'sex','type':'discrete', 'values':[0,1]}]
xfunc(tdf, datasplits=datasplits, cols=['col'])