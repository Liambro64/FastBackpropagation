#include "../Project.hpp"

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<ddd> dis(-1, 1);

ddd randD()
{
	return dis(gen);
}

NetworkTrainer::NetworkTrainer(int inputs, vec<int> layers) : inputs(inputs), network(inputs, layers, randD), Layers(layers) {}



void NetworkTrainer::Load(const std::string &fileName, vec<vec<ddd>> (*f)(std::ifstream *, int), int maxLines)
{
	std::cout << "Loading data from " << fileName << std::endl;
	std::ifstream file(fileName);
	if (!file.is_open())
	{
		std::cerr << "Error opening file: " << fileName << std::endl;
		return;
	}
	data = f(&file, maxLines);
	std::cout << "Loaded " << data.size() << " Lines of data from " << fileName << std::endl;
}
vec<ddd> NetworkTrainer::Train(vec<ddd> (*formatExpectedOutput)(vec<ddd>, vec<ddd>), int epochs, ddd learningRate, int datapoints, int printAfter)
{
	if (data.empty())
	{
		std::cerr << "No data loaded. Please load data before training." << std::endl;
		return {};
	}
	if (datapoints <= 0 || datapoints > data.size())
		throw std::invalid_argument("Load more data.");
	if (printAfter < 0)
		printAfter = 1;
	network.alpha = learningRate;
	std::cout << "Training network with " << epochs << " epochs and learning rate " << network.alpha << " and a data size of " << data.size() << std::endl << std::endl << std::endl << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	vec<ddd> errorGradientVector(0);
	auto last = start;
	int trendSize = printAfter;
	int batches = this->inputs / data[0].size();
	double trend = 0;
	vec<ddd> trendVector(0);
	for (int epoch = 0; epoch < epochs; epoch++)
	{
		auto startEpoch = std::chrono::high_resolution_clock::now();
		if (epoch != 0)
			network.alpha = learningRate; 
		double err = 0;
		std::cout << "Epoch " << epoch + 1 << "/" << epochs << std::endl << std::endl << std::endl;
		double lastError = 0;
		for (int i = 0; i < datapoints; i++)
		{
			double currentError = 0;
			std::vector<double> inputs(this->inputs);
			std::vector<double> out;
			std::vector<double> expectedOut(Layers[Layers.size() - 1]);
			for (int j = 0; j < batches; j++)
			{
				for (int k = 0; k < data[i].size(); k++) {
				inputs[(j * data[i].size()) + k] = data[i + j][k];
				}
			}
			expectedOut = formatExpectedOutput(data[i+batches], data[i+(batches - 1)]);
			err += (currentError = network.Learn(inputs, expectedOut));
			if (std::isnan(currentError))
			{
				std::cerr << "Became nan. inspect, and press anything to continue" << std::endl;
				getchar();
			}
			// std::cout << "Current Error: " << currentError << " for datapoint " << i << " for epoch " << epoch+1 << std::endl;
			if (printAfter != 0 && i % printAfter == 0)
			{
				bool is0 = (i == 0);
				if (is0)
					i++;
				errorGradientVector.push_back(currentError);
				double percent = ((100 * ((double)epoch * (double)datapoints + i)) / (double)(epochs * datapoints));
				std::string progressBar = std::to_string(percent);
				int j = 0;
				progressBar += "% ";
				while (j <= percent)
				{
					progressBar += "█";
					j++;
				}
				if (percent - (int)percent >= 0.75) {
					progressBar += "#";
					j++;
				} else if (percent - (int)percent >= 0.5) {
					progressBar += "|";
					j++;
				} else if (percent - (int)percent >= 0.25) {
					progressBar += "/";
					j++;
				}
				while (j < 100)
				{
					progressBar += "-";
					j++;
				}
				auto now = std::chrono::high_resolution_clock::now();
				auto totalduration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
				auto epochduration = std::chrono::duration_cast<std::chrono::milliseconds>(now - startEpoch).count();
				auto lastduration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count();
				int64_t estimatedTotalDuration = (totalduration * epochs * datapoints) / (epoch * datapoints + i);
				int64_t estimatedTimeLeft = estimatedTotalDuration - totalduration;
				std::cout << "\x1b[1F\x1b[2K\x1b[1F\x1b[2K" << "Datapoint " << i << "/" << datapoints << ""
						  << ", \tCurrent Error: " << currentError << ",\tTrend: " << trend 
						  << ", \tTime for this epoch: " << millisToString(epochduration) << ", \tTotal Time: " << millisToString(totalduration)
						  << ", \tEstimated Time left: " << millisToString(estimatedTimeLeft)  << "\n" << progressBar << ", \tEstimated Total Time: " << millisToString(estimatedTotalDuration) << std::endl;
				last = now;
				if (is0)
					i--;
				
			}
			if (trendVector.size() < trendSize)
			{
				trendVector.push_back(currentError - lastError);
				trend += currentError - lastError;
			}
			else
			{
				trend -= trendVector[0];
				for (int i = 0; i < trendVector.size() - 1; i++)
				{
					trendVector[i] = trendVector[i + 1];
				}
				trendVector[trendVector.size() - 1] = currentError - lastError;
				trend += currentError - lastError;
			}
			lastError = currentError;
		}
		err /= datapoints;
		std::cout << "\x1b[1F\x1b[2K\x1b[1F\x1b[2K\x1b[1F\x1b[2K" << "Epoch " << epoch + 1 << "/" << epochs << " complete, Average Error: " << err << std::endl;
	}
	return errorGradientVector;
}

vec<ddd> NetworkTrainer::TrainGPU(int epochs, ddd learningRate, int datapoints, int printAfter)
{
	if (data.empty())
	{
		std::cerr << "No data loaded. Please load data before training." << std::endl;
		return {};
	}
	if (printAfter < 0)
		printAfter = 1;
	network.alpha = learningRate;
	std::cout << "Training network with " << epochs << " epochs and learning rate " << network.alpha << " and a data size of " << data.size() << std::endl << std::endl << std::endl << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	vec<ddd> errorGradientVector(0);
	auto last = start;
	for (int epoch = 0; epoch < epochs; epoch++)
	{
		auto startEpoch = std::chrono::high_resolution_clock::now();
		if (epoch != 0)
			network.alpha = learningRate / (epoch + 1); // Decrease learning rate over time for better convergence
		double err = 0;
		std::cout << "\x1b[1F\x1b[2K\x1b[1F\x1b[2K\x1b[1F\x1b[2K" << "Epoch " << epoch + 1 << "/" << epochs << std::endl << std::endl << std::endl;
		for (int i = 0; i < datapoints; i++)
		{
			double currentError = 0;
			std::vector<double> inputs(Layers[0]);
			std::vector<double> out;
			std::vector<double> expectedOut(Layers[Layers.size() - 1]);
			for (int j = 0; j < Layers[0] / data[i].size(); j++)
			{
				inputs[(j * data[i].size())] = data[i + j][0];
				inputs[(j * data[i].size()) + 1] = data[i + j][1];
			}

			expectedOut[0] = data[i + Layers[0] / data[i].size()][0];
			err += (currentError = network.LearnGPU(inputs, expectedOut, learningRate));
			if (std::isnan(currentError))
			{
				std::cerr << "Became nan. inspect, and press anything to continue" << std::endl;
				getchar();
			}
			// std::cout << "Current Error: " << currentError << " for datapoint " << i << " for epoch " << epoch+1 << std::endl;
			if (printAfter != 0 && i % printAfter == 0)
			{
				bool is0 = (i == 0);
				if (is0)
					i++;
				errorGradientVector.push_back(currentError);
				double percent = ((100 * ((double)epoch * (double)datapoints + i)) / (double)(epochs * datapoints));
				std::string progressBar = std::to_string(percent);
				int j = 0;
				progressBar += "% ";
				while (j <= percent)
				{
					progressBar += "█";
					j++;
				}
				if (percent - (int)percent >= 0.75) {
					progressBar += "#";
					j++;
				} else if (percent - (int)percent >= 0.5) {
					progressBar += "|";
					j++;
				} else if (percent - (int)percent >= 0.25) {
					progressBar += "/";
					j++;
				}
				while (j < 100)
				{
					progressBar += "-";
					j++;
				}
				auto now = std::chrono::high_resolution_clock::now();
				auto totalduration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
				auto epochduration = std::chrono::duration_cast<std::chrono::milliseconds>(now - startEpoch).count();
				auto lastduration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count();
				double estimatedTotalDuration = ((double)totalduration) / (percent/100);
				double estimatedTimeLeft = estimatedTotalDuration - totalduration;
				int days = estimatedTimeLeft / (1000 * 60 * 60 * 24);
				int hours = (int)(estimatedTimeLeft / (1000 * 60 * 60)) % 24;
				int minutes = (int)(estimatedTimeLeft / (1000 * 60)) % 60;
				int seconds = (int)(estimatedTimeLeft / 1000) % 60;
				std::cout << "\x1b[1F\x1b[2K\x1b[1F\x1b[2K" << "Datapoint " << i << "/" << datapoints
						  << ", \tAverage Error: " << err / (i + 1) 
						  << ", \tTime for this epoch: " << epochduration / 1000 << "s,\tTotal Time: " << totalduration / 1000 << "s, \tTime since last: " << lastduration / 1000 << "s"
						  << ", \tEstimated Time left: " << millisToString(estimatedTimeLeft) << std::endl << progressBar << " Estimated Total Time: "<<std::endl;
				last = now;
				if (is0)
					i--;
			}
		}
		err /= datapoints;
		std::cout << "Epoch " << epoch + 1 << "/" << epochs << " complete, Average Error: " << err << std::endl;
	}
	return errorGradientVector;
}
size_t NetworkTrainer::SaveWeights(std::string filename){
	network.SaveWeights(filename);
}
bool NetworkTrainer::LoadWeights(std::string filename){
	network.LoadWeights(filename);
}
vec<ddd> NetworkTrainer::RunGPU(vec<ddd> *input)
{
	return network.RunGPU(input);
}