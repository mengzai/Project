#!/usr/bin/env bash

python predict_by_rf.py # 预测

paste -d "," data/task1_uids data/xxtemp__0324 > QCfWsd_rf0324.csv # 结果
/bin/rm data/xxtemp__0324 # 删临时文件

