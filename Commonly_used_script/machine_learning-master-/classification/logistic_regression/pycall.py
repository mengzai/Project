import ctypes
from ctypes import *

ll = ctypes.cdll.LoadLibrary 
lib = ll("./liblr.so") 

lib.TrainLR.restype = c_int # function return type is int
lib.TrainLR.argtypes = [c_int, c_int, c_void_p, c_void_p, c_void_p, c_int, c_float, c_int, c_char_p] # function arguments types
lib.TestLR.restype = c_int
lib.TestLR.argtypes = [c_char_p, c_int, c_int, c_void_p, c_void_p]

n = 4                  # num of samples
d = 2                  # dimensions
w = (c_float*n)()      # weights of samples
X = (c_float*(n*d))()  # attributes of samples
y = (c_int*n)()        # labels of samples
opti = 4               # option
lamb = 0.01            # lambda
verb = 0               # display optimization or not
save = c_char_p("model.txt") # save model

y_py = [1, 1, -1, -1];
X_py = [1, 0, 0, 1, -1, 0, 0, -1];
w_py = [1, 1, 1, 1];

# copy values from python lists into ctypes
for i in range(0, n):
	y[i] = y_py[i]
	w[i] = w_py[i]

for i in range(0,len(X)):
	X[i] = X_py[i]

# copy values from python lists into ctypes
for i in range(0, n):
    y[i] = y_py[i]
    w[i] = w_py[i]

for i in range(0, len(X)):
    X[i] = X_py[i]

# call c function
lib.TrainLR(n, d, w, X, y, opti, lamb, verb, save);

f = (c_float*n)()      # output f(x) = w'*x+b
lib.TestLR(save, n, d, X, f); # return 0 success
for i in range(0, len(f)):
	print f[i]

# http://www.cnblogs.com/night-ride-depart/p/4907613.html