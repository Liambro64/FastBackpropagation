main = main.cpp
dbgout = bin/dbg.out
out = bin/main.out
code = src/dataPuller.cpp src/Acceleration.cu src/NeuralNetwork.cpp src/NetworkTrainer.cpp src/Math.cpp

all : runtest

runtest : buildtest
	./$(out)

buildtest :
	nvcc -arch=compute_86 -l curl $(main) $(code) -o $(out)

rundebug :
	./$(dbgout)

debugbuild :
	nvcc -g -l curl $(main) $(code) -o $(dbgout)