train_gbdt,test_gbdt:
	g++ -Wno-logical-op-parentheses -O2 -o train_gbdt train_gbdt.cpp gbrt.cpp -I ../../data -I ../../optimization ../../data/data.cpp
	g++ -Wno-logical-op-parentheses -O2 -o test_gbdt  test_gbdt.cpp  gbrt.cpp -I ../../data -I ../../optimization ../../data/data.cpp
clean:
	rm -f *.o train_gbdt test_gbdt
