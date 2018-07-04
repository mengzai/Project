train_cart,test_cart:
	g++ -Wno-logical-op-parentheses -O2 -o train_cart train_cart.cpp cart.cpp -I ../../data -I ../../optimization ../../data/data.cpp
	g++ -Wno-logical-op-parentheses -O2 -o test_cart test_cart.cpp cart.cpp -I ../../data -I ../../optimization ../../data/data.cpp
clean:
	rm -f *.o train_cart test_cart
