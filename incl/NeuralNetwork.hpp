#pragma once
#ifndef NEURALNETWORK_HPP
# define NEURALNETWORK_HPP
# include "../Project.hpp"
class NeuralNetwork {
private:
	//How the weights will be set out:
	//First vector holds a vector of vectors (goto next line)
	//Second vector holds a vector of vectors (goto next line)
	//Third vector holds a vector of doubles set out as the following:
	//	[0] = the neuron's bias
	//	[1] = the weight for the input from the first neuron of the previous layer
	//	[2] = the weight for the input from the second neuron of the previous layer
	//and so on
	//hopefully this will make it easier to calculate because they are all somewhat close
	vec<vec<vec<ddd>>> weights;
	int inputs;
	ddd (*randFunc)();
public:

	ddd alpha = 0.01; //learning rate, default value
	//constructor
	NeuralNetwork(int inputs, vec<int> layerSizes, ddd randFunc());
	//destructor
	~NeuralNetwork() {};
	vec<vec<ddd>>		extractBiases();
	vec<vec<vec<ddd>>>	extractWeights();

	void				InjectBiases(const vec<vec<ddd>>& extractedBiases);
	void				InjectWeights(const vec<vec<vec<ddd>>>& extractedWeights);

	//function to get the weights of the neural network
	sptr<vec<vec<vec<ddd>>>> getWeights();
	//function to calculate the output of the neural network given an input vector
	vec<ddd> Run(vec<ddd> *input);
	vec<ddd> RunGPU(vec<ddd> *input);
	ddd Learn(vec<ddd> input, vec<ddd> expectedOutput);
	ddd LearnGPU(vec<ddd> input, vec<ddd> expectedOutput, ddd learningRate);
	size_t SaveWeights(std::string filename);
	bool LoadWeights(std::string filename);
};

#endif