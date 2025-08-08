main = main.cpp
dbgout = bin/dbg.out
out = bin/main.out
code = src/Acceleration.cu src/NeuralNetwork.cpp src/NetworkTrainer.cpp src/Math.cpp
all : bin runtest

bin: 
	mkdir bin

dll :
	g++ -shared -o bin/libNeuralNetwork.dll src/NeuralNetwork.cpp src/NetworkTrainer.cpp src/Math.cpp -fPIC

runtest : buildtest
	./$(out)

buildtest :
	nvcc -arch=compute_86 -l curl $(main) $(code) -o $(out)

rundebug :
	./$(dbgout)

debugbuild :
	nvcc -g -l curl $(main) $(code) -o $(dbgout)
	
runStreamTest : buildStreamTest
	bin/streamtest.out
buildStreamTest :
	g++ tmp.cpp src/CommandPiper.cpp -o bin/streamtest.out