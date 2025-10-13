#pragma once
#ifndef NETWORKTRAINER_HPP
# define NETWORKTRAINER_HPP

# include "../Project.hpp"
class NetworkTrainer {
private:
	NeuralNetwork network;
	std::vector<int> Layers;
	int inputs;
public:
	NetworkTrainer(int inputs, std::vector<int> layers);

	~NetworkTrainer() = default;
	std::vector<std::vector<ddd>> data;
	void Load(const std::string& fileName, std::vector<std::vector<ddd>> (*f)(std::ifstream *, int), int maxLines = -1);
	//returns a gradient of doubles which represents the error of the network over time
	std::vector<ddd> Train(std::vector<ddd> (*formatExpectedOutput)(std::vector<ddd>, std::vector<ddd>), int epochs = 1000, ddd learningRate = 0.01, int datapoints = 1000, int printAfter = -1);
	std::vector<ddd> TrainOffFunctions(std::vector<ddd> (*input)(std::vector<ddd>), std::vector<ddd> (*output)(std::vector<ddd>), int epochs = 100, ddd learningRate = 0.0001, int datapoints = 1000000);
	//std::vector<ddd> TrainGPU(int epochs = 1000, double learningRate = 0.01, int datapoints = 1000, int printAfter = -1);
	std::vector<ddd> Run(std::vector<ddd> (*formatExpectedOutput)(std::vector<ddd>, std::vector<ddd>), int datapoints);
	NeuralNetwork getNetwork() {return network;}
	//std::vector<ddd> RunGPU(std::vector<ddd> *input) ;
	size_t SaveWeights(std::string filename);
	bool LoadWeights(std::string filename);
};
#endif