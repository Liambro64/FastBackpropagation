#pragma once
#ifndef NETWORKTRAINER_HPP
# define NETWORKTRAINER_HPP

# include "../Project.hpp"
class NetworkTrainer {
private:
	NeuralNetwork network;
	vec<int> Layers;
	int inputs;
public:
	NetworkTrainer(int inputs, vec<int> layers);

	~NetworkTrainer() = default;
	std::vector<std::vector<double>> data;
	void Load(const std::string& fileName, vec<vec<ddd>> (*f)(std::ifstream *, int), int maxLines = -1);
	//returns a gradient of doubles which represents the error of the network over time
	vec<ddd> Train(vec<ddd> (*formatExpectedOutput)(vec<ddd>, vec<ddd>), int epochs = 1000, double learningRate = 0.01, int datapoints = 1000, int printAfter = -1);
	vec<ddd> TrainGPU(int epochs = 1000, double learningRate = 0.01, int datapoints = 1000, int printAfter = -1);
	vec<ddd> RunGPU(vec<ddd> *input) ;
	size_t SaveWeights(std::string filename);
	bool LoadWeights(std::string filename);
};
#endif