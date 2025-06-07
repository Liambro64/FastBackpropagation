main = main.cpp
dbgout = bin/dbg.out
out = bin/main.out
code = src/NeuralNetwork.cpp src/NetworkTrainer.cpp

all : runtest

runtest : buildtest
	./$(out)

buildtest :
	g++ $(main) $(code) -o $(out)

rundebug :
	./$(out)

debugbuild :
	g++ -g $(main) $(code) -o $(dbgout)