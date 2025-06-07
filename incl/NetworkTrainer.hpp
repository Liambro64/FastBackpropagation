#pragma once
#ifndef NETWORKTRAINER_HPP
#define NETWORKTRAINER_HPP

#include "../Project.hpp"
class NetworkTrainer {
private:
	NeuralNetwork network;
	vec<int> Layers;
public:
	NetworkTrainer(vec<int> layers);

	~NetworkTrainer() = default;
	std::vector<std::vector<double>> data;
	void Load(const std::string& fileName, int maxLines = -1);
	void Train(int epochs = 1000, double learningRate = 0.01, int datapoints = 1000);
};
#endif