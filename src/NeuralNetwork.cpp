#include "../Project.hpp"

ddd NeuralNetwork::sigmoid(ddd x)
{
	return 1.0 / (1.0 + exp(-x));
}
ddd NeuralNetwork::sigmoidInverse(ddd x)
{
	return std::log(x / (1 - x));
}

NeuralNetwork::NeuralNetwork(vec<int> layerSizes, ddd (*randFunc)()) : randFunc(randFunc)
{
	weights = vec<vec<vec<ddd>>>();
	// Initialize weights for each layer
	for (int i = 0; i < layerSizes.size(); i++)
	{
		int prevSize = (i == 0) ? 0 : layerSizes[i - 1];
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
	vec<ddd> lastVals(input->size());
	std::copy(input->begin(), input->end(), lastVals.begin());
	for (int i = 1; i < weights.size(); i++)
	{
		vec<ddd> vals(weights[i].size());
		for (int j = 0; j < weights[i].size(); j++)
		{
			vals[j] = weights[i][j][0];
			for (int k = 1; k < weights[i][j].size(); k++)
			{
				vals[j] += lastVals[k - 1] * weights[i][j][k];
			}
			vals[j] = sigmoid(vals[j]);
		}
		lastVals = vals;
	}
	return lastVals;
}
ddd NeuralNetwork::Learn(vec<ddd> *input, vec<ddd> *expectedOutput, ddd learningRate)
{
	// run but keep the values
	vec<vec<ddd>> lastVals(weights.size());
	std::copy(input->begin(), input->end(), lastVals[0].begin());
	for (int i = 1; i < weights.size(); i++)
	{
		vec<ddd> vals(weights[i].size());
		for (int j = 0; j < weights[i].size(); j++)
		{
			vals[j] = weights[i][j][0];
			for (int k = 1; k < weights[i][j].size(); k++)
			{
				vals[j] += lastVals[i - 1][k - 1] * weights[i][j][k];
			}
			vals[j] = sigmoid(vals[j]);
		}
		lastVals[i] = vals;
	}
	vec<vec<ddd>> errVals(weights.size());
	vec<vec<ddd>> trueVals(weights.size());
	for (int i = 0; i < weights.size(); i++)
	{
		errVals[i] = vec<ddd>(weights[i].size());
		trueVals[i] = vec<ddd>(weights[i].size());
	}
	for (int i = lastVals.size() - 1; i > 0; i--)
	{
		for (int j = 0; j < lastVals[i].size(); j++)
		{
			if (i == lastVals.size() - 1)
			{
				trueVals[i][j] = sigmoidInverse(lastVals[i][j]);
				errVals[i][j] = expectedOutput->at(j) - lastVals[i][j];
			}
			else
			{
				trueVals[i][j] = sigmoidInverse(lastVals[i][j]);
				for (int k = 0; k < errVals[i + 1].size(); k++)
				{
					// the error of the "neuron" should increase by the error of its connections
					// although the neuron's synapses have already been altered
					errVals[i][j] += ((trueVals[i][j] * weights[i + 1][k][j + 1]) / trueVals[i + 1][k]);
				}
			}
			for (int k = 0; k < lastVals[i - 1].size(); k++)
			{
				// this should change the weights accordingly
				weights[i][j][k + 1] *= 1 + alpha * (((sigmoidInverse(lastVals[i - 1][k]) * weights[i][j][k + 1]) / lastVals[i][j]) * errVals[i][j]);
			}
			weights[i][j][0] *= 1 + alpha * (weights[i][j][0] / sigmoidInverse(lastVals[i][j]));
		}
	}
	ddd totErr = 0;
	for (int i = 0; i < errVals.size(); i++) {
		for (int j = 0; j < errVals[i].size(); j++) {
			totErr = errVals[i][j];
		}
	}
	return totErr;
}