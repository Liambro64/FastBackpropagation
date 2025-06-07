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
	//there will be a second vector of vectors for each layer's neuron's values
	ddd sigmoid(ddd x);
	ddd sigmoidInverse(ddd x);
	ddd (*randFunc)();
public:
	ddd alpha = 0.01; //learning rate, default value
	//constructor
	NeuralNetwork(vec<int> layerSizes, ddd randFunc());
	//destructor
	~NeuralNetwork() {};
	//function to get the weights of the neural network
	sptr<vec<vec<vec<ddd>>>> getWeights();
	//function to calculate the output of the neural network given an input vector
	vec<ddd> Run(vec<ddd> *input);
	ddd Learn(vec<ddd> *input, vec<ddd> *expectedOutput, ddd learningRate);
};

#endif