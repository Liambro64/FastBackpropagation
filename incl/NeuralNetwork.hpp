#pragma once
#ifndef NEURALNETWORK_HPP
# define NEURALNETWORK_HPP
# include "../Project.hpp"
class NeuralNetwork {
private:
	//How the weights will be set out:
	//First std::vectortor holds a std::vectortor of std::vectortors (goto next line)
	//Second std::vectortor holds a std::vectortor of std::vectortors (goto next line)
	//Third std::vectortor holds a std::vectortor of doubles set out as the following:
	//	[0] = the neuron's bias
	//	[1] = the weight for the input from the first neuron of the previous layer
	//	[2] = the weight for the input from the second neuron of the previous layer
	//and so on
	//hopefully this will make it easier to calculate because they are all somewhat close
	std::vector<std::vector<std::vector<ddd>>> weights;
	int inputs;
	std::vector<int> layerSizes;
	std::vector<ddd *> dev_weights;
	ddd (*randFunc)();
public:

	ddd alpha = 0.01; //learning rate, default value
	NeuralNetwork() {}
	bool operator==(const NeuralNetwork& other);
	//constructor
	NeuralNetwork(int inputs, std::vector<int> layerSizes, ddd randFunc());
	//destructor
	~NeuralNetwork() {
		if (dev_weights.size() > 0) {
			FreeWeightsGPU(dev_weights);
		}
	};
	std::vector<std::vector<ddd>>		extractBiases();
	std::vector<std::vector<std::vector<ddd>>>	extractWeights();

	void				InjectBiases(const std::vector<std::vector<ddd>>& extractedBiases);
	void				InjectWeights(const std::vector<std::vector<std::vector<ddd>>>& extractedWeights);

	//function to get the weights of the neural network
	sptr<std::vector<std::vector<std::vector<ddd>>>> getWeights();
	//function to calculate the output of the neural network given an input std::vectortor
	std::vector<ddd> Run(std::vector<ddd> *input);
	void GenerateWeights(int inputs, std::vector<int> layerSizes);
	void AllocateGPUWeights();
	ddd LearnGPU(std::vector<ddd> input, std::vector<ddd> expectedOutput);
	long SetWeightsFromArray(std::vector<ddd> weightsAsArray);
	vec<vec<vec<ddd>>> getWeughts() { return weights; }
	std::vector<ddd> RunGPU(std::vector<ddd> *input);
	ddd Learn(std::vector<ddd> input, std::vector<ddd> expectedOutput);
	//ddd LearnGPU(std::vector<ddd> input, std::vector<ddd> expectedOutput, ddd learningRate); not functional yet
	size_t SaveWeights(std::string filename);
	bool LoadWeights(std::string filename);
};

#endif