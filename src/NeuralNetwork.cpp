#include "../Project.hpp"

std::vector<std::vector<std::vector<ddd>>> NeuralNetwork::extractWeights()
{
	std::vector<std::vector<std::vector<ddd>>> extractedWeights;
	for (int i = 0; i < weights.size(); i++)
	{
		std::vector<std::vector<ddd>> extractedLayer;
		for (int j = 0; j < weights[i].size(); j++)
		{
			std::vector<ddd> extractedNeuron(weights[i][j].size() - 1);									// Exclude bias
			std::copy(weights[i][j].begin() + 1, weights[i][j].end(), extractedNeuron.begin()); // Copy weights excluding bias
			extractedLayer.push_back(extractedNeuron);
		}
		extractedWeights.push_back(extractedLayer);
	}
	return extractedWeights;
}
void NeuralNetwork::InjectWeights(const std::vector<std::vector<std::vector<ddd>>> &extractedWeights)
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
bool NeuralNetwork::operator==(const NeuralNetwork &other)
{
	bool equal = true; // (inputs == other.inputs && weights == other.weights);
	try
	{
		if (equal)
		{
			for (int i = 0; i < weights.size(); i++)
			{
				for (int j = 0; j < weights[i].size(); j++)
				{
					for (int k = 0; k < weights[i][j].size(); k++)
					{
						if (weights[i][j][k] != other.weights[i][j][k])
						{
							std::cout << "Weights differ at layer " << i << ", neuron " << j << ", weight " << k << std::endl;
							return false; // If any weight is different, return false
						}
					}
				}
			}
			return true;
		}
	}
	catch (const std::exception &e)
	{
		std::cout << "Error comparing NeuralNetworks: " << e.what() << std::endl;
	}
	return false;
}
std::vector<std::vector<ddd>> NeuralNetwork::extractBiases()
{
	std::vector<std::vector<ddd>> extractedWeights;
	for (int i = 0; i < weights.size(); i++)
	{
		std::vector<ddd> extractedLayer;
		for (int j = 0; j < weights[i].size(); j++)
		{
			extractedLayer.push_back(weights[i][j][0]);
		}
		extractedWeights.push_back(extractedLayer);
	}
	return extractedWeights;
}
void NeuralNetwork::InjectBiases(const std::vector<std::vector<ddd>> &extractedBiases)
{
	for (int i = 0; i < weights.size(); i++)
	{
		for (int j = 0; j < weights[i].size(); j++)
		{
			weights[i][j][0] = extractedBiases[i][j]; // Bias is always at index 0
		}
	}
}
NeuralNetwork::NeuralNetwork(int inputs, std::vector<int> layerSizes, ddd (*randFunc)()) : randFunc(randFunc), inputs(inputs)
{
	weights = std::vector<std::vector<std::vector<ddd>>>();
	// Initialize weights for each layer
	for (int i = 0; i < layerSizes.size(); i++)
	{
		int prevSize = (i == 0) ? inputs : layerSizes[i - 1];
		std::vector<std::vector<ddd>> layerWeights;
		for (int j = 0; j < layerSizes[i]; j++)
		{
			std::vector<ddd> neuronWeights;
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
void NeuralNetwork::GenerateWeights(int inputs, std::vector<int> layerSizes)
{
	weights = std::vector<std::vector<std::vector<ddd>>>();
	// Initialize weights for each layer
	for (int i = 0; i < layerSizes.size(); i++)
	{
		int prevSize = (i == 0) ? inputs : layerSizes[i - 1];
		std::vector<std::vector<ddd>> layerWeights;
		for (int j = 0; j < layerSizes[i]; j++)
		{
			std::vector<ddd> neuronWeights;
			// Bias for the neuron
			neuronWeights.push_back(0);
			// Weights for inputs from previous layer neurons
			for (int k = 0; k < prevSize; k++)
			{
				neuronWeights.push_back(0);
			}
			layerWeights.push_back(neuronWeights);
		}
		weights.push_back(layerWeights);
	}
}
long NeuralNetwork::SetWeightsFromArray(std::vector<ddd> weightsAsArray)
{
	long totalIndex = 0;
	for (int i = 0; i < weights.size(); i++)
	{
		for (int j = 0; j < weights[i].size(); j++)
		{
			for (int k = 0; k < weights[i][j].size(); k++)
			{
				weights[i][j][k] = weightsAsArray[totalIndex++];
			}
		}
	}
	return totalIndex;
}

std::vector<ddd> NeuralNetwork::Run(std::vector<ddd> *input)
{
	return NetworkRunSum(*input, weights);
}
// std::vector<ddd> NeuralNetwork::RunGPU(std::vector<ddd> *input)
// {
// 	return FullRun(*input, weights)[weights.size() - 1];
// }
ddd NeuralNetwork::Learn(std::vector<ddd> input, std::vector<ddd> expectedOutput)
{
	// run but keep the values
	std::vector<std::vector<ddd>> values;
	values.resize(weights.size() + 1);
	values[0] = input;
	for (int i = 0; i < weights.size(); i++)
	{
		values[i + 1] = weightedSums(values[i], weights[i]);
		if (values[i + 1].size() == 0)
		{
			throw std::runtime_error("errored at: " + i);
		}
	}
	double loss = LossFunction(values[weights.size()], expectedOutput);
	std::vector<std::vector<ddd>> errVals(weights.size());			  // deltas (for biases)
	std::vector<std::vector<std::vector<ddd>>> weightChanges(weights.size()); // deltas (for weights)
	for (int i = 0; i < weights.size(); i++)
	{
		errVals[i].resize(weights[i].size());
	}
	for (int j = 0; j < weights[weights.size() - 1].size(); j++)
	{
		errVals[weights.size() - 1][j] = LossDerivative(expectedOutput[j], values[weights.size()][j]) * sigmoidDerivative(values[weights.size()][j]);
	}
	auto preTransposedWeights = transpose(extractWeights());
	for (int i = weights.size() - 1; i >= 0; i--)
	{
		weightChanges[i] = outerProduct(errVals[i], values[i]);
		if (i != 0)
		{
			std::vector<ddd> err = vector_matrix_multiply(errVals[i], preTransposedWeights[i]);
			for (int j = 0; j < weights[i - 1].size(); j++)
			{
				errVals[i - 1][j] = err[j] * sigmoidDerivative(values[i][j]);
			}
		}
	}
	for (int i = 0; i < weights.size(); i++)
	{
		for (int j = 0; j < weights[i].size(); j++)
		{
			weights[i][j][0] += alpha * errVals[i][j];
			for (int k = 1; k < weights[i][j].size(); k++)
			{
				weights[i][j][k] += alpha * weightChanges[i][j][k - 1];
			}
		}
	}

	return loss;
}

// ddd NeuralNetwork::LearnGPU(std::vector<ddd> input, std::vector<ddd> expectedOutput, ddd learningRate)
// {
// }
size_t NeuralNetwork::SaveWeights(std::string filename)
{
	try
	{
		std::string size = "";
		std::fstream stream(filename);
		
		if (stream.is_open() == false)
			throw std::invalid_argument("Couldnt load file into stream");
		for (int i = 0; i < weights.size(); i++)
		{
			for (int j = 0; j < weights[i].size(); j++)
			{
				for (int k = 0; k < weights[i][j].size(); k++)
				{
					size.append(DoubleToUnreadableString(&(weights[i][j][k])));
				}
			}
		}
		std::string sizeSize = std::to_string(size.size());
		sizeSize.append(std::to_string(weights.size()));
		sizeSize.append(" ");
		sizeSize.append(std::to_string(inputs));
		for (int i = 0; i < weights.size(); i++)
		{
			sizeSize.append(" ");
			sizeSize.append(std::to_string(weights[i].size()));
		}
		sizeSize.append("\n");
		stream.write(sizeSize.data(), sizeSize.size());
		stream.write(size.data(), size.size());
		stream.close();
		return size.size();
	}
	catch (const std::exception &e)
	{
		std::cerr << "Error saving weights: " << e.what() << std::endl;
		return -1;
	}
	return 0;
}
bool NeuralNetwork::LoadWeights(std::string filename)
{
	try
	{
		std::ifstream stream(filename);
		if (stream.is_open() == false)
		{
			throw std::invalid_argument("Couldnt load file into stream");
		}
		std::string size;
		size.resize(2048);
		stream.getline(size.data(), 2048, '\n');
		std::string main;
		long lsize = 0;
		std::vector<int> sizes = {};
		long tmplyrSize = 0;
		int i;
		for (i = 0; i < size.size(); i++)
		{
			if (size[i] < '0' || size[i] > '9')
				break;
			lsize = lsize * 10 + (size[i] - '0');
		}
		for (i++; i < size.size(); i++)
		{
			if (size[i] < '0' || size[i] > '9')
				break;
			tmplyrSize = tmplyrSize * 10 + (size[i] - '0');
		}
		inputs = tmplyrSize;
		tmplyrSize = 0;
		for (i++; i < size.length(); i++)
		{
			if (size[i] < '0' || size[i] > '9')
			{
				if (tmplyrSize == 0)
					continue;
				sizes.push_back(tmplyrSize);
				tmplyrSize = 0;
				continue;
			}
			tmplyrSize = tmplyrSize * 10 + (size[i] - '0');
		}
		GenerateWeights(inputs, sizes);

		main.resize(lsize + 1);
		stream.read(main.data(), lsize);
		std::vector<ddd> weightsAsArray = longUnreadableStringToArray(main);
		SetWeightsFromArray(weightsAsArray);
		// praying this works
		return true;
	}
	catch (const std::exception &e)
	{
		std::cerr << "Error loading weights: " << e.what() << std::endl;
		return false;
	}
	return false;
}