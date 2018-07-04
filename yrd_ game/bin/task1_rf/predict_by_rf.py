#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

import sys
import os

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def get_use_gbm_label():
    res = {}
    cntset = dict()
    for line in open('data/ignore_f_map_os_ip'): # keep_fname_sp74
        if line.strip() == "":
            continue
        cntset[line.strip()] = True
    linenum = 0
    for line in open('data/f_map_add_os_ip'): # f_map_224
        if line.strip() == "":
            continue
        tmp = line.strip()
        if tmp not in cntset:
            res[linenum] = tmp
        linenum += 1
        cntset[tmp] = False
    print "feature_cnt:", len(res)
    return res

def clf_func(Xtrain, ytrain, Xtest, ytest, load_model=''):
    clf = rf_clf
    if load_model:
        modelfile = open(load_model, 'rb')
        modeldumps = ''.join(modelfile.readlines())
        clf = pickle.loads(modeldumps)
    else:
        clf.fit(Xtrain, ytrain)
    preds = clf.predict_proba(Xtest)
    scores = [round(p[1], 6) for p in preds]
    return clf, scores

rf_clf = RandomForestClassifier(max_depth=4, max_features=0.3,
            min_samples_leaf=20, min_samples_split=20,
            n_estimators=150, random_state=None,
            n_jobs=12, oob_score=False)

print rf_clf

def typefunc(num):
    num = num.strip()
    if not num:
        return -0.0001
    try:
        return int(num)
    except:
        try:
            return float(num)
        except:
            raise ValueError("Not num type: " + str(num))
            return num

def load_gbm_data(fn, labelindex=0, split="\t"):
    fealist, labellist = [], []
    linenum = 0
    for line in open(fn):
        if line.strip() == "":
            continue
        linenum += 1
        line = line.strip().split(split)
        try:
            label = float(line[labelindex].strip())
            float(line[-1])
        except:
            continue
        features = map(typefunc, line[labelindex + 1:])
        fealist.append(features)
        labellist.append(label)
    return np.array(fealist), np.array(labellist)

def main():
    Xtrain, ytrain = load_gbm_data(trainfile)
    usefeat = get_use_gbm_label()
    indexs = sorted(usefeat)
    if indexs:
        Xtrain = Xtrain[:, indexs]
    Xtest, ytest = load_gbm_data(testfile)
    if indexs:
        Xtest = Xtest[:, indexs]
    load_model1 = 'data/task1_rf.model7' + "_bak"
    load_model2 = 'data/task1_rf.model9' + "_bak"
    clf, scores1 = clf_func(Xtrain, ytrain, Xtest, [], load_model1)
    clf, scores2 = clf_func(Xtrain, ytrain, Xtest, [], load_model2)
    outf = 'data/xxtemp__0324'
    uidf = 'data/task1_uids'
    scoref = 'QCfWsd_rf0324.csv'
    of = open(outf, 'w')
    of.write("predict\n")
    for s in range(len(scores1)):
        pred = 0.5 * (scores1[s] + scores2[s])
        of.write(str(pred) + "\n")
    return None

if __name__ == '__main__':
    trainfile = 'data/211_train_vali_mobile_ip_os_device'
    testfile = 'data/211_test_mobile_ip_2200'
    main()
