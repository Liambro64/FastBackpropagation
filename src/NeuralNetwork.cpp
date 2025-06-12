#include "../Project.hpp"

vec<vec<vec<ddd>>> NeuralNetwork::extractWeights()
{
	vec<vec<vec<ddd>>> extractedWeights;
	for (int i = 0; i < weights.size(); i++)
	{
		vec<vec<ddd>> extractedLayer;
		for (int j = 0; j < weights[i].size(); j++)
		{
			vec<ddd> extractedNeuron(weights[i][j].size() - 1); // Exclude bias
			std::copy(weights[i][j].begin() + 1, weights[i][j].end(), extractedNeuron.begin()); // Copy weights excluding bias
			extractedLayer.push_back(extractedNeuron);
		}
		extractedWeights.push_back(extractedLayer);
	}
	return extractedWeights;
}
void NeuralNetwork::InjectWeights(const vec<vec<vec<ddd>>> &extractedWeights)
{
	for (int i = 0; i < weights.size(); i++)
	{
		for (int j = 0; j < weights[i].size(); j++)
		{
			for (int k = 1; k < weights[i][j].size(); k++)
			{
				weights[i][j][k] = extractedWeights[i][j][k - 1]; // Adjust index for bias
			}
		}
	}
}

vec<vec<ddd>> NeuralNetwork::extractBiases()
{
	vec<vec<ddd>> extractedWeights;
	for (int i = 0; i < weights.size(); i++)
	{
		vec<ddd> extractedLayer;
		for (int j = 0; j < weights[i].size(); j++)
		{
			extractedLayer.push_back(weights[i][j][0]);
		}
		extractedWeights.push_back(extractedLayer);
	}
	return extractedWeights;
}
void NeuralNetwork::InjectBiases(const vec<vec<ddd>> &extractedBiases)
{
	for (int i = 0; i < weights.size(); i++)
	{
		for (int j = 0; j < weights[i].size(); j++)
		{
			weights[i][j][0] = extractedBiases[i][j]; // Bias is always at index 0
		}
	}
}
NeuralNetwork::NeuralNetwork(int inputs, vec<int> layerSizes, ddd (*randFunc)()) : randFunc(randFunc), inputs(inputs)
{
	weights = vec<vec<vec<ddd>>>();
	// Initialize weights for each layer
	for (int i = 0; i < layerSizes.size(); i++)
	{
		int prevSize = (i == 0) ? inputs : layerSizes[i - 1];
		vec<vec<ddd>> layerWeights;
		for (int j = 0; j < layerSizes[i]; j++)
		{
			vec<ddd> neuronWeights;
			// Bias for the neuron
			neuronWeights.push_back(randFunc());
			// Weights for inputs from previous layer neurons
			for (int k = 0; k < prevSize; k++)
			{
				neuronWeights.push_back(randFunc());
			}
			layerWeights.push_back(neuronWeights);
		}
		weights.push_back(layerWeights);
	}
}
vec<ddd> NeuralNetwork::Run(vec<ddd> *input)
{
	return NetworkRunSum(*input, weights);
}
vec<ddd> NeuralNetwork::RunGPU(vec<ddd> *input)
{
	return FullRun(*input, weights)[weights.size() - 1];
}
ddd NeuralNetwork::Learn(vec<ddd> input, vec<ddd> expectedOutput, ddd learningRate)
{
	// run but keep the values
	vec<vec<ddd>> values;
	values.resize(weights.size());
	for (int i = 0; i < weights.size(); i++)
	{
		values[i] = weightedSums(i == 0 ? input : values[i - 1], weights[i]);
		if (values[i].size() == 0)
		{
			throw std::runtime_error("errored at: " + i);
		}
	}
	double loss = LossFunction(expectedOutput, values[weights.size() - 1]);
	vec<vec<ddd>> errVals(weights.size());			  // deltas (for biases)
	vec<vec<vec<ddd>>> weightChanges(weights.size()); // deltas (for weights)
	for (int i = 0; i < weights.size(); i++)
	{
		errVals[i].resize(weights[i].size());
	}
	for (int j = 0; j < weights[weights.size() - 1].size(); j++)
	{
		errVals[weights.size() - 1][j] = LossDerivative(expectedOutput[j], values[weights.size() - 1][j]) * sigmoidDerivative(values[weights.size() - 1][j]);
	}
	auto preTransposedWeights = transpose(extractWeights());
	for (int i = weights.size() - 2; i >= 0; i--)
	{
		weightChanges[i] = outerProduct(values[i], errVals[i + 1]);
		if (i != 0)
		{
			vec<ddd> err = vector_matrix_multiply(errVals[i + 1], preTransposedWeights[i + 1]);
			for (int j = 0; j < weights[i].size(); j++)
			{
				errVals[i][j] = err[j] * sigmoidDerivative(values[i][j]);
			}
		}
	}
	for (int i = 1; i < weightChanges.size(); i++)
	{
		for (int j = 0; j < weightChanges[i].size(); j++)
		{
			weights[i][j][0] -= learningRate * errVals[i][j];
			for (int k = 1; k < weights[i][j].size(); k++)
			{
				weights[i][j][k] -= learningRate * weightChanges[i-1][k - 1][j];
			}
		}
	}

	return loss;
}

ddd NeuralNetwork::LearnGPU(vec<ddd> input, vec<ddd> expectedOutput, ddd learningRate)
{
	
}
