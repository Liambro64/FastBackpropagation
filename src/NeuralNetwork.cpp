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
ddd NeuralNetwork::Learn(vec<ddd> input, vec<ddd> expectedOutput)
{
	// run but keep the values
	vec<vec<ddd>> values;
	values.resize(weights.size() + 1);
	values[0] = input;
	for (int i = 0; i < weights.size(); i++)
	{
		values[i+1] = weightedSums(values[i], weights[i]);
		if (values[i+1].size() == 0)
		{
			throw std::runtime_error("errored at: " + i);
		}
	}
	double loss = LossFunction(values[weights.size()], expectedOutput);
	vec<vec<ddd>> errVals(weights.size());			  // deltas (for biases)
	vec<vec<vec<ddd>>> weightChanges(weights.size()); // deltas (for weights)
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
		weightChanges[i] = outerProduct(values[i], errVals[i]);
		if (i != 0)
		{
			vec<ddd> err = vector_matrix_multiply(errVals[i], preTransposedWeights[i]);
			for (int j = 0; j < weights[i].size(); j++)
			{
				errVals[i - 1][j] = err[j] * sigmoidDerivative(values[i][j]);
			}
		}
	}
	for (int i = 0; i < weights.size(); i++)
	{
		for (int j = 0; j < weights[i].size(); j++)
		{
			weights[i][j][0] -= alpha * errVals[i][j];
			for (int k = 1; k < weights[i][j].size(); k++)
			{
				weights[i][j][k] -= alpha * weightChanges[i][k - 1][j];
			}
		}
	}

	return loss;
}

ddd NeuralNetwork::LearnGPU(vec<ddd> input, vec<ddd> expectedOutput, ddd learningRate)
{
	
}
size_t NeuralNetwork::SaveWeights(std::string filename) {
	std::string size = "";
	std::fstream stream(filename);
	if (stream.is_open() == false)
		throw std::invalid_argument("Couldnt load file into stream");
	
	size.append(std::to_string(weights.size()));
	size.append(" ");
	for (int i = 0; i < weights.size(); i++) {
		size.append(std::to_string(weights[i].size()));
		size.append(";");
		for (int j = 0; j < weights[i].size(); j++) {
			size.append(std::to_string(weights[i][j].size()));
			size.append(":");
			for (int k = 0; k < weights[i][j].size(); k++) {
				size.append(std::to_string(weights[i][j][k]));
				size.append(",");
			}
			size.append(":");
		}
		size.append(";");
	}
	size.append(" ");
	size.append(std::to_string(inputs));
	std::string sizeSize = std::to_string(size.size());
	sizeSize.append("\n");
	stream.write(sizeSize.data(), sizeSize.size());
	stream.write(size.data(), size.size());
	stream.close();
	return size.size();
}
bool NeuralNetwork::LoadWeights(std::string filename) {
	std::ifstream stream(filename);
	if (stream.is_open() == false)
		throw std::invalid_argument("Couldnt load file into stream");
	std::string size;
	size.resize(128);
	stream.getline(size.data(), 128, '\n');
	std::string main;
	main.resize((long)std::strtod(size.data(), nullptr));
	stream.getline(main.data(), main.size());
	//from here, split into different parts
	// " " | ";" | ":" | ","
	vec<std::string> size_layer_input = split(main, ' ');
	weights.resize(strtol(size_layer_input[0].data(), nullptr, 10));
	inputs = std::strtol(size_layer_input[2].data(), nullptr, 10);
	for (int i = 0; i < weights.size(); i++) {
		vec<std::string> firstLayers = split(size_layer_input[1], ';');
		weights[i].resize(std::strtol(firstLayers[i*2].data(), nullptr, 10));
		for (int j = 0; j < weights[i].size(); i++) {
			vec<std::string> secondLayers = split(firstLayers[i*2 + 1], ':');
			weights[i][j].resize(std::strtol(secondLayers[j*2].data(), nullptr, 10));
			vec<std::string> finalVals = split(secondLayers[j*2+1], ',');
			for (int k = 0; k < weights[i][j].size(); k++) {
				weights[i][j][k] = strtod(finalVals[k].data(), nullptr);
			}
		}
	}
	//praying this works
	return true;
}