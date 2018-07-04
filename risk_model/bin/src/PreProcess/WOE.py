import json
import pandas as pd
import numpy as np
from Utils.Log import Log


class WOE:
	def __init__(self, preprocess_config):
		with open(preprocess_config.bins_file) as f:
			self.bins = json.load(f)
		self.woe = dict()
		self.all_odds = ""
	
	def _get_woe(self, bad_good_cnt):
		return np.log((bad_good_cnt[0]/bad_good_cnt[1])/self.all_odds)

	def fit(self, label, features):
		column_names = set(features.columns)
		#Get woe
		for col_name in self.bins:	
			#left close and right open
			bad_good_cnt = [[1e-10,1e-10] for i in range(0, len(self.bins[col_name])-1)]
			if col_name not in column_names:
				Log.Warning("No such feature was named: " + col_name)
				continue
			for row_idx, value in enumerate(features[col_name]):
				for bin_idx, bins_split_point in enumerate(self.bins[col_name][1:]):
					if value < bins_split_point:
						break
				if label[row_idx] == 1:
					bad_good_cnt[bin_idx][1] += 1
				else:
					bad_good_cnt[bin_idx][0] += 1
			if self.all_odds == "":
				all_bad_good_cnt = reduce(lambda x,y: [x[0]+y[0], x[1]+y[1]], bad_good_cnt)
				self.all_odds = all_bad_good_cnt[0]/all_bad_good_cnt[1] 
			self.woe[col_name] = map(lambda x: self._get_woe(x), bad_good_cnt) 
	
	def transform(self, features):
		for col_name in self.woe:	
			woe_value = []
			for row_idx, value in enumerate(features[col_name]):
				for bin_idx, bins_split_point in enumerate(self.bins[col_name][1:]):
					if value < bins_split_point:
						break
				woe_value.append(self.woe[col_name][bin_idx])
			features[col_name] = woe_value
