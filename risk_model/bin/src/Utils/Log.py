class Log:
	log_level = 0
	log_param = {"debug":-1, "info":0, "warning":1, "fatal":2}
	
	@staticmethod
	def Debug(message):
		if Log.log_param["debug"] >= Log.log_level: 
			print("[PIPE] [Debug] " + message) 

	@staticmethod
	def Info(message):
		if Log.log_param["info"] >= Log.log_level: 
			print("[PIPE] [Info] " + message) 

	@staticmethod
	def Warning(message):
		if Log.log_param["warning"] >= Log.log_level: 
			print("[PIPE] [Warning] " + message) 

	@staticmethod
	def Fatal(message):
		if Log.log_param["fatal"] >= Log.log_level: 
			print("[PIPE] [Fatal] " + message) 
		raise Exception(message)
	
	@staticmethod
	def set_level(log_level = 0):
		Log.log_level = log_level
