train_svm,test_svm:
	g++ -Wno-logical-op-parentheses -O2 -o train_svm train_svm.cpp svm.cpp -I ../../data -I ../../optimization ../../data/data.cpp ../../optimization/lbfgs.cpp -L. lbfgsb.so
	g++ -Wno-logical-op-parentheses -O2 -o test_svm test_svm.cpp svm.cpp -I ../../data -I ../../optimization ../../data/data.cpp ../../optimization/lbfgs.cpp -L. lbfgsb.so
clean:
	rm -f *.o train_svm test_svm
