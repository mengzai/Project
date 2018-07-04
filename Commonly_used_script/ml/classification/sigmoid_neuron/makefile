train_sn,test_sn:
	g++ -Wno-logical-op-parentheses -O2 -o train_sn train_sn.cpp sn.cpp ../logistic_regression/lr.cpp -I ../../data -I ../../optimization ../../data/data.cpp ../../optimization/lbfgs.cpp -L. lbfgsb.so
	g++ -Wno-logical-op-parentheses -O2 -o test_sn test_sn.cpp sn.cpp ../logistic_regression/lr.cpp -I ../../data -I ../../optimization ../../data/data.cpp ../../optimization/lbfgs.cpp -L. lbfgsb.so
clean:
	rm -f *.o train_sn test_sn
