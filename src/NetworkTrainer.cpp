#include "../Project.hpp"

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<ddd> dis(-1, 1);

ddd randD()
{
	return dis(gen);
}

NetworkTrainer::NetworkTrainer(vec<int> layers) : network(layers, randD), Layers(layers) {}

void NetworkTrainer::Load(const std::string &fileName, int maxLines)
{
	std::cout << "Loading data from " << fileName << std::endl;
	std::ifstream file(fileName);
	if (!file.is_open())
	{
		std::cerr << "Error opening file: " << fileName << std::endl;
		return;
	}
	data.clear();
	std::string line;
	int i = 0;
	while (std::getline(file, line))
	{
		std::vector<double> row;
		std::string high = line.substr(25, 7);
		std::string low = line.substr(33, 7);
		row.push_back(std::stod(high));
		row.push_back(std::stod(low));
		data.push_back(row);
		if (maxLines != -1 && ++i >= maxLines)
		{
			break; // Stop reading if maxLines is reached
		}
	}
	std::cout << "Loaded Data from " << fileName << std::endl;
}
void NetworkTrainer::Train(int epochs, ddd learningRate, int datapoints)
{
	
	if (data.empty()) {
		std::cerr << "No data loaded. Please load data before training." << std::endl;
		return;
	}
	network.alpha = learningRate;
	std::cout << "Training network with " << epochs << " epochs and learning rate " << network.alpha << " and a data size of " << data.size() << std::endl;
	for (int epoch = 0; epoch < epochs; epoch++) {
		if (epoch != 0)
			network.alpha = learningRate / (epoch + 1); // Decrease learning rate over time for better convergence
		double err = 0;
		for (int i = 0; i < datapoints; i++) {
			double currentError = 0;
			std::vector<double> inputs(Layers[0]);
			std::vector<double> out;
			std::vector<double> expectedOut(Layers[Layers.size() - 1]);
			for (int j = 0; j < Layers[0] / data[i].size(); j++) {
				inputs[(j*4)] = data[i+j][0]; 
				inputs[(j*4)+1] = data[i+j][1];
			}
			
			expectedOut[0] = data[i + Layers[0] / data[i].size()][0] > data[i + Layers[0] / data[i].size() - 1][0]
						? 1 : data[i + Layers[0] / data[i].size()][0] == data[i + Layers[0] / data[i].size() - 1][0]
						? 0 : -1;
			err += (currentError = network.Learn(&inputs, &expectedOut, learningRate));
			std::cout << "Current Error: " << currentError << " for datapoint " << i << " for epoch " << epoch+1 << std::endl;
		}
		err /= datapoints;
		std::cout << "Epoch " << epoch + 1 << "/" << epochs << ", Average Error: " << err << std::endl;
	}
}
