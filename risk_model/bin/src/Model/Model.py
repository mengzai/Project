#!/usr/bin/plabelthon
# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

class Model(object):
	def __init__(self, model_params):
		self.params = model_params
		self.method = model_params.method
		if self.method == 'xgb':
			self.model = Xgboost(self.params.xgb_params)
		if self.method == 'lr':
			self.model = Lr(self.params.lr_params)

	def train(self, features, label, weight=None):
		if self.method == 'lr':
			self.model.train(features, label, weight)
		elif self.method == 'xgb':
			self.model.train(features, label, weight)
			print type(self.model).__name__
		return self.model

	def predict_proba(self, features):
		try:
			if self.method == 'xgb':
				return self.model.predict(features)
			elif self.method == 'lr':
				return self.model.predict_proba(features)[:,1]
		except ValueError:
			print 'please train the model first'

	def predict(self, features):
		try:
			return self.model.predict(features)
		except ValueError:
			print 'please train the model first'


class Lr(object):
	def __init__(self, lr_params):
		self.params = {}
		self.params['penalty'] = lr_params.penalty
		self.params['dual'] = lr_params.dual
		self.params['tol'] = lr_params.tol
		self.params['C'] = lr_params.C
		self.params['fit_intercept'] = lr_params.fit_intercept
		self.params['intercept_scaling'] = lr_params.intercept_scaling
		self.params['class_weight'] = lr_params.class_weight
		self.params['random_state'] = lr_params.random_state
		self.params['solver'] = lr_params.solver
		self.params['max_iter'] = lr_params.max_iter
		self.params['multi_class'] = lr_params.multi_class
		self.params['verbose'] = lr_params.verbose
		self.params['warm_start'] = lr_params.warm_start
		self.params['n_jobs'] = lr_params.n_jobs
	
		self.lr_model

	def train(self, features, label, weight):
		self.lr_model = LogisticRegression(**self.params)
		self.lr_model.fit(features, label, sample_weight=weight)

	def predict_proba(self, features):
		try:
			return self.lr_model.predict_proba(features)
		except ValueError:
			print 'please train the model first'

	def predict(self, features):
		try:
			return self.lr_model.predict(features)
		except ValueError:
			print 'please train the model first'


class Xgboost(object):
	def __init__(self, xgb_params):
		self.params = {}
		self.params['max_depth'] = xgb_params.max_depth
		self.params['eta'] = xgb_params.eta
		self.params['n_estimators'] = xgb_params.n_estimators
		self.params['silent'] = xgb_params.silent
		self.params['objective'] = xgb_params.objective
		self.params['eval_metric'] = xgb_params.eval_metric
		self.params['nthread'] = xgb_params.nthread
		self.params['gamma'] = xgb_params.gamma
		self.params['min_child_weight'] = xgb_params.min_child_weight
		self.params['max_delta_step'] = xgb_params.max_delta_step
		self.params['subsample'] = xgb_params.subsample
		self.params['colsample_bytree'] = xgb_params.colsample_bytree
		self.params['alpha'] = xgb_params.alpha
		self.params['reg_lambda'] = xgb_params.reg_lambda
		self.params['scale_pos_weight'] = xgb_params.scale_pos_weight
		self.params['base_score'] = xgb_params.base_score
		self.params['seed'] = xgb_params.seed
		##train parameters
		self.evals= xgb_params.evals
		self.obj = xgb_params.obj
		self.feval= xgb_params.feval
		self.learning_rates= xgb_params.learning_rates
		self.early_stopping_rounds = xgb_params.early_stopping_rounds
		self.num_boost_round = xgb_params.num_boost_round

		self.xgb_model = ""

	def train(self, features, label, weight):
		dtrain = xgb.DMatrix(features, label=label, weight=weight, feature_names=features.columns)
		watchlist = [(dtrain, 'train')]
		self.xgb_model = xgb.train(
			self.params,
			dtrain,
			evals=watchlist,
			obj=self.obj,
			feval=self.feval,
			learning_rates=self.learning_rates,
			num_boost_round=self.num_boost_round,
			early_stopping_rounds=self.early_stopping_rounds)

	def predict(self, features):
		try:
			test_dmatrix = xgb.DMatrix(features, feature_names=features.columns)
			return self.xgb_model.predict(test_dmatrix)
		except ValueError:
			print 'please train the model first'
