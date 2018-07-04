import numpy as np
import pandas as pd
import copy

from Utils.Log import Log

def GetStatistic(data):
	sep_dict = {'comma':0, 'tab':0, 'colon':0, 'space':0}
	for i in data:
		if i == ',':
			sep_dict['comma'] += 1
		elif i == '\t':
			sep_dict['tab'] += 1
		elif i == ':':
			sep_dict['colon'] += 1
		elif i == ' ':
			sep_dict['space'] += 1
	trans_dict = {'comma':',', 'tab':'\t', 'colon':':', 'space':' '}
	return trans_dict[sorted(sep_dict.items(), key=lambda x:x[1], reverse=True)[0][0]]



class DataLoader:
	def __init__(self, loader_config):
		self.header = loader_config.header
		self.ignored_col = loader_config.ignored_col
		self.missing_value = loader_config.missing_value
		self.label_col = loader_config.label_col
		self.weight_col = loader_config.weight_col
		self.train_file = loader_config.train_file
		self.test_file = loader_config.test_file

		self.name2idx_ = dict()
		self.ignored_set_ = set()
		self.used_idx_list_ = list()
		self.label_idx_ = ""
		self.weight_idx_ = ""
		self.heads = ""

		self.data = Data()

	def _set_header(self, header=""):
		if self.header == True:
			sep = GetStatistic(header)
			#Change feature names to index
			for idx, name in enumerate(header.strip().split(sep)):
				self.name2idx_[name] = idx

		#Get label index
		if self.label_col == "":
			Log.Fatal("Need label col!")
		elif "name:" in self.label_col:
			label_name = self.label_col[5:]
			self.label_idx_ = self.name2idx_[label_name] 
			Log.Info("Set " + label_name + " as label")
		else:
			self.label_idx_ = int(self.label_col)
			Log.Info("Set col " + self.label_col + " as label")
		self.ignored_set_.add(self.label_idx_)

		#Get weight index
		if self.weight_col == "":
			pass
		else:
			if "name:" in self.weight_col:
				weight_name = self.weight_col[5:]
				self.weight_idx_ = self.name2idx_[self.weight_col[5:]] 
				Log.Info("Set " + weight_name + " as weight")
			else:
				self.weight_idx_ = int(self.weight_col)
				Log.Info("Set col " + self.weight_col + " as weight")
			self.ignored_set_.add(self.weight_idx_)

		#Set ignored col
		if self.ignored_col == "":
			pass
		elif "name:" in self.ignored_col:
			for name in self.ignored_col[5:].strip().split(','):
				self.ignored_set_.add(self.name2idx_[name])
			Log.Info("Set " + self.ignored_col[5:] + " as ignored col")
		else:
			for idx in self.ignored_col.strip().split(','):
				self.ignored_set_.add(int(idx))
			Log.Info("Set col" + self.ignored_col + " as ignored col")
	

	#Parse csv, tsc, ssv format file
	def sv_parser(self, line, sep):
		"""
		if len(self.used_idx_list_) == 0:
			self.used_idx_list_ = sorted(list(set(range(0, len(line.split(',')))) - self.ignored_set_))
		value = ""
		one_sample_features = []
		col_idx = 0
		used_idx = 0
		for ch in line:
			if ch == sep:
				if col_idx != self.used_idx_list_[used_idx]:
					col_idx += 1
					continue
				else:
					one_sample_features.append(self._float(value))
					value = ""
					col_idx += 1
					used_idx += 1
			else:
				value += ch

		"""
		data = line.strip().split(sep)
		if len(self.used_idx_list_) == 0:
			self.used_idx_list_ = sorted(list(set(range(0, len(data))) - self.ignored_set_))
		self.data.features.append(map(lambda x: self.missing_value if data[x] == "" or data[x].lower() == "null" else data[x], self.used_idx_list_))
		#self.data.features.append(map(lambda x: self._float(data[x]), self.used_idx_list_))
		self.data.label.append(float(data[self.label_idx_]))
		if self.weight_idx_ != "":
			self.data.weight.append(float(data[self.weight_idx_]))


	def load_data(self, filename):
		self.data.clear()
		Log.Info("Loading Data: "+filename)

		#Get the separator
		with open(filename, 'r') as f:
			if self.header == True and self.label_idx_ == "":
				self._set_header(f.readline())
			elif self.label_idx_ == "":
				self._set_header()
			line1 = f.readline()
			line2 = f.readline()
			sep1 = GetStatistic(line1)
			sep2 = GetStatistic(line2)
		if (sep1 == sep2):
			sep = sep1
		else:
			Log.Fatal("Unkown separator!")

		#Start to loading data
		with open(filename, 'r') as f:
			if self.header == True:
				f.readline()
			for line in f:
				#To do: parse libsvm format data
				if sep == ':':
					pass
				else:
					self.sv_parser(line, sep)
					#self.sv_parser(f, sep)

		#If theres is no original header, set idx header, which will convenient for pre-processing
		#Or just use original header
		if self.header == False:
			self.data.heads = list(set(range(0, len(self.data.features[0])+len(self.ignored_set_))) - self.ignored_set_)
		else:
			self.data.heads = map(lambda x: x[0], \
									filter(lambda y: y[1] not in self.ignored_set_, \
										sorted(self.name2idx_.items(), key=lambda x:x[1], reverse=False)
									)
								 )
		#Trans lists to DataFrame
		self.data.trans_2_df()
		return copy.deepcopy(self.data)

class Data:
	def __init__(self):
		self.name2idx_ = dict()
		self.label = []
		self.features = []
		self.weight = []
		self.heads = ""
	
	def trans_2_df(self):
		self.features = pd.DataFrame(self.features, columns=self.heads, dtype="float64")
		self.label = np.array(self.label)
		self.weight = np.array(self.weight)
	
	def clear(self):
		self.name2idx_ = dict()
		self.label = []
		self.features = []
		self.weight = []
		self.heads = ""
