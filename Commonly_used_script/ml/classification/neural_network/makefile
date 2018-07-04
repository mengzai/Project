train_svm,test_svm:
	g++ -Wno-logical-op-parentheses -O2 -o train_nn train_nn.cpp nn.cpp shuffle.cpp -I ../../data -I ../../optimization ../../data/data.cpp
	g++ -Wno-logical-op-parentheses -O2 -o test_nn  test_nn.cpp  nn.cpp shuffle.cpp -I ../../data -I ../../optimization ../../data/data.cpp
clean:
	rm -f *.o train_nn test_nn
