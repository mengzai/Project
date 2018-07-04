from Utils.Log import Log
class ConfigBase:
	#Because variable in Python cannot be referenced, there is a default value. 
	def get_bool(self, var_name, params, default_value):
		if var_name in params:
			if params[var_name].lower() == "false":
				return False
			elif params[var_name].lower() == "true":
				return True
			else:
				Log.Fatal(var_name+" should be Boolean Type, and now it is "+params[var_name])
		else:
			return default_value
	
	def get_float(self, var_name, params, default_value):
		if var_name in params:
			try:
				return float(params[var_name])
			except:
				Log.Fatal(var_name+" should be Float Type, and now it is "+params[var_name])
		else:
			return default_value

	def get_int(self, var_name, params, default_value):
		if var_name in params:
			try:
				return int(params[var_name])
			except:
				Log.Fatal(var_name+" should be Int Type, and now it is "+params[var_name])
		else:
			return default_value
	
	def get_str(self, var_name, params, default_value):
		if var_name in params:
			try:
				return str(params[var_name])
			except:
				Log.Fatal(var_name+" should be String Type, and now it is "+params[var_name])
		else:
			return default_value

class ConfigLoader(ConfigBase):
	def __init__(self, filename, delimiter='='):
		self.delimiter=delimiter
		self.params = dict()
		self.loader_params = LoaderParams()
		self.preprocess_params = PrePocessParams()
		self.model_params = ModelBaseParams()
		self._str_2_dict(filename)

	def _str_2_dict(self, filename):
		with open(filename,'r') as f:
			for line in f:
				if '=' not in line:
					continue
				pos = line.find('#')
				if pos == -1:
					key, value = line.strip().split(self.delimiter)
					self.params[key.strip()] = value.strip()
				elif pos != 0:
					key, value = line[:pos].strip().split(self.delimiter)
					self.params[key.strip()] = value.strip()
	
	def set(self):
		self.loader_params.set(self.params)
		self.preprocess_params.set(self.params)
		self.model_params.set(self.params)
			
class LoaderParams(ConfigBase):
	def __init__(self):
		self.header = False
		self.ignored_col = ""
		self.missing_value = -1
		self.label_col = ""
		self.weight_col = ""
		self.train_file = ""
		self.test_file = ""

	def set(self, params):
		self.header = self.get_bool("header", params, False)
		self.missing_value = self.get_float("missing_value", params, -1)
		self.label_col = self.get_str("label_col", params, "")
		self.ignored_col = self.get_str("ignored_col", params, "")
		self.weight_col = self.get_str("weight_col", params, "")
		self.train_file = self.get_str("train", params, "")
		self.test_file = self.get_str("test", params, "")

class PrePocessParams(ConfigBase):
	def __init__(self):
		self.bins_file = ""
	def set(self, params):
		self.bins_file = self.get_str("bins_file", params, "")
		print self.bins_file

class ModelBaseParams(ConfigBase):
	def __init__(self):
		self.method = ""
	def set(self, params):
		self.method = self.get_str("method", params, "").lower()
		if self.method == "":
			Log.Fatal("Need set method in (Lr, Xgb)")
		if self.method == "lr":
			self.lr_params = LrParams()
			self.lr_params.set(params)
		elif self.method == "xgb":
			self.xgb_params = XgbParams()
			self.xgb_params.set(params)
	
class LrParams(ConfigBase):
	def set(self, params):
		self.penalty = self.get_str("penalty", params, "l2")
		self.dual = self.get_bool("dual", params, False)
		self.tol = self.get_float("tol", params, 1e-4)
		self.C = self.get_float("C", params, 1.0)
		self.fit_intercept = self.get_bool("fit_intercept", params, True)
		self.intercept_scaling = self.get_float("intercept_scaling", params, 1.0)
		#To do get dict
		#self.class_weight = self.get_dict("class_weight", params, None)
		self.class_weight = "balanced"
		self.random_state = self.get_int("random_state", params, None)
		self.solver = self.get_str("solver", params, "liblinear")
		self.max_iter = self.get_int("max_iter", params, 100)
		self.multi_class = self.get_str("multi_class", params, "ovr")
		self.verbose = self.get_int("verbose", params, 0)
		self.warm_start = self.get_bool("warm_start", params, False)
		self.n_jobs = self.get_int("n_jobs", params, 1)

class XgbParams(ConfigBase):
	def set(self, params):
		self.max_depth = self.get_int("max_depth", params, 6)
		self.eta = self.get_float("eta", params, 0.3)
		self.n_estimators = self.get_int("n_estimators", params, 100)
		self.silent = self.get_int("silent", params, 0)
		self.objective = self.get_str("objective", params, "reg:linear")
		self.eval_metric = self.get_str("eval_metric", params, "rmse")
		self.nthread = self.get_int("thread", params, 8)
		self.gamma = self.get_float("gamma", params, 0)
		self.min_child_weight = self.get_float("min_child_weight", params, 1)
		self.max_delta_step = self.get_float("max_delta_step", params, 0)
		self.subsample = self.get_float("subsample", params, 1)
		self.colsample_bytree = self.get_float("colsample_bytree", params, 1)
		self.alpha = self.get_float("alpha", params, 0)
		self.reg_lambda = self.get_float("lambda", params, 1)
		self.scale_pos_weight = self.get_float("scale_pos_weight", params, 1)
		self.base_score = self.get_float("base_score", params, 0.5)
		self.seed = self.get_int("seed", params, 1024)
		##train parameters
		self.obj = None
		self.feval= None 
		self.evals=None
		self.learning_rates = None
		self.early_stopping_rounds = None 
		self.num_boost_round = self.get_int("num_boost_round", params, 100)
