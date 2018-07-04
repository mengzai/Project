#!/usr/bin/python
# -*- coding: utf-8 -*-
from Utils.Log import Log
from IO.ConfigLoader import ConfigLoader  
from IO.DataLoader import DataLoader
from PreProcess.WOE import WOE
from Model.Model import Model
from Metric.Metric import Metric

import time
import argparse
import numpy as np
import os
parser = argparse.ArgumentParser()
parser.add_argument('--config',type=str,default='config')
args = parser.parse_args()

Log.set_level(-1)
config_loader = ConfigLoader(args.config)
config_loader.set()

data_loader=DataLoader(config_loader.loader_params)
s=time.time()
trdata = data_loader.load_data(data_loader.train_file)
tedata = data_loader.load_data(data_loader.test_file)
print tedata.features.shape
print trdata.features.shape
e=time.time()
Log.Debug("It costs " + str(e-s) + "s")
Log.Info("Finish load data")

Log.Info("Start training")
model = Model(config_loader.model_params)
model.train(trdata.features, np.ravel(trdata.label))
te_prob = model.predict_proba(tedata.features)

mec = Metric(tedata.label, te_prob)
Log.Info("AUC: " + str(mec.calcAuc()))
Log.Info("KS: " + str(mec.calcKs()))
