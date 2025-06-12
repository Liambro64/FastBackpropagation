main = main.cpp
dbgout = bin/dbg.out
out = bin/main.out
code = src/Acceleration.cu src/NeuralNetwork.cpp src/NetworkTrainer.cpp src/Math.cpp 

all : runtest

runtest : buildtest
	./$(out)

buildtest :
	nvcc -arch=compute_86 $(main) $(code) -o $(out)

rundebug :
	./$(dbgout)

debugbuild :
	nvcc -g $(main) $(code) -o $(dbgout)