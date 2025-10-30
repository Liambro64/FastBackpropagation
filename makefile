main = main.cpp
dbgout = bin/dbg.out
out = bin/main.out
code = src/Acceleration.cu src/NeuralNetwork.cpp src/NetworkTrainer.cpp src/Math.cpp
all : bin runtest

bin: 
	mkdir bin

so :
	g++ -shared -o bin/libNeuralNetwork.so src/NeuralNetwork.cpp src/NetworkTrainer.cpp src/Math.cpp -fPIC

runtest : buildtest
	./$(out)

buildtest :
	nvcc -arch=compute_86 $(main) $(code) -o $(out)

rundebug : debugbuild
	./$(dbgout)

debugbuild :
	nvcc -g $(main) $(code) -o $(dbgout)
	