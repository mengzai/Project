import numpy as np
import pandas as pd

def load_data(filename):
	return pd.read_csv(filename, error_bad_lines=False)

def outliers_detection(data, times = 7, quantile = 0.95):
    data=data[-data.isnull()]
    data = np.array(sorted(data))
    #std-outlier
    outlier1 = np.mean(data) + 1*np.std(data)

    # mad-outlier
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    outlier2 = med + times * mad

    # quantile-outlier
    outlier3 = data[int(np.floor(quantile * len(data)) - 1)]
    return outlier1, outlier2, outlier3

if __name__ == '__main__':
	#d = [1,4,5,64,2,-1,5,23,128,3,4,2,43,123,4111,1,100000]
	traindata = load_data('xxd_good_and_m7')
	feature = "MAX_LOAN_AMOUNT"
	outlier1, outlier2, outlier3=outliers_detection(traindata[feature])
	print outlier1, outlier2, outlier3

