train_lr,test_lr:
	g++ -Wno-logical-op-parentheses -O2 -o train_lr train_lr.cpp lr.cpp -I ../../data -I ../../optimization ../../data/data.cpp ../../optimization/lbfgs.cpp -L. lbfgsb.so
	g++ -Wno-logical-op-parentheses -O2 -o test_lr test_lr.cpp lr.cpp -I ../../data -I ../../optimization ../../data/data.cpp ../../optimization/lbfgs.cpp -L. lbfgsb.so
clean:
	rm -f *.o train_lr test_lr
