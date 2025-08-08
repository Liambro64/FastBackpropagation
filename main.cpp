#include "Project.hpp"

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<ddd> dis(-1, 1);
double randomFunc()
{
	return dis(gen);
}
void NetworkTest()
{
	
	vec<int> layers = {400, 250, 120, 40, 12, 6};
	NeuralNetwork network = NeuralNetwork(8, layers, randomFunc);
	vec<ddd> ins(4);
	for (int i = 0; i < 4; i++)
		ins[i] = randomFunc();
	network.Run(&ins);
}

void TrainerTest()
{
	vec<int> layers = {500, 350, 200, 120, 60, 20, 6};
	NetworkTrainer Trainer = NetworkTrainer(8, layers);
	Trainer.Load("Data/Stock/AUDUSD/Data15M.csv", &formatAUDUSDData, 25600);
	Trainer.Train(&formatExpectedOutputAUDUSDCurrent, 50, 0.05, 25000, 25);
}
void smalltrainertest()
{
	vec<int> layers = {350, 200, 135, 90, 60, 20, 6};
	int data = 5000;
	int epochs = 400;
	int nextPo2 = (int)std::pow(2, std::ceil(std::log2(data)));
	NetworkTrainer Trainer = NetworkTrainer(8, layers);
	Trainer.Load("Data/Stock/AUDUSD/Data15M.csv", &formatAUDUSDData, nextPo2);
	Trainer.Train(&formatExpectedOutputAUDUSDCurrent, 400, 0.4, 5000, 50);
}

// not working
// void largeTrainertestGPU()
// {
// 	vec<int> layers = {50, 125, 200, 345, 500, 415, 255, 135, 90, 60, 20, 1};
// 	NetworkTrainer Trainer = NetworkTrainer(8, layers);
// 	Trainer.Load("Data/Stock/AUDUSD/Data15M.csv", &formatAUDUSDData, 25600);

// 	Trainer.TrainGPU(10, 0.4, 25000);
// }

void largedatatrainertest()
{
	vec<int> layers = {90, 25, 6};
	int data = 50000;
	int epochs = 400;
	int nextPo2 = (int)std::pow(2, std::ceil(std::log2(data)));
	NetworkTrainer Trainer = NetworkTrainer(8, layers);
	Trainer.Load("Data/Stock/AUDUSD/Data15M.csv", &formatAUDUSDData, nextPo2);
	Trainer.Train(&formatExpectedOutputAUDUSDCurrent, 400, 0.4, data, 500);
}
NetworkTrainer saveCPUTest(int data = 50000, int epochs = 20)
{
	vec<int> layers = {350, 200, 135, 90, 60, 20, 5};
	int inputs = 20;
	int nextPo2 = (int)std::pow(2, std::ceil(std::log2(data)) + 1);
	NetworkTrainer Trainer = NetworkTrainer(inputs, layers);
	Trainer.Load("Data/Stock/AUDUSD/Data15M.csv", &formatAUDUSDData, nextPo2);
	Trainer.Train(&formatExpectedOutputAUDUSDCurrent, epochs, 0.02, data, 25);
	Trainer.SaveWeights("WeightsSaves/WeightsRCT.fbp");
	return Trainer;
}
NetworkTrainer CPULoadTest(int data = 50000, int epochs = 25, vec<int> layers = {500, 200, 70, 30, 5}, int printafter = 25)
{
	int inputs = 20;
	int nextPo2 = (int)std::pow(2, std::ceil(std::log2(data)) + 1);
	NetworkTrainer Trainer = NetworkTrainer(inputs, layers);
	Trainer.Load("Data/Stock/AUDUSD/Data15M.csv", &formatAUDUSDData, nextPo2);
	Trainer.Train(&formatExpectedOutputAUDUSDCurrent, epochs, 0.02, data, printafter);
	return Trainer;
}
NetworkTrainer weightLoadCPUTest(int data = 50000)
{
	vec<int> layers = {350, 200, 135, 90, 60, 20, 5};
	int inputs = 20;
	int nextPo2 = (int)std::pow(2, std::ceil(std::log2(data)) + 1);
	NetworkTrainer Trainer = NetworkTrainer(inputs, layers);
	Trainer.Load("Data/Stock/AUDUSD/Data15M.csv", &formatAUDUSDData, nextPo2);
	Trainer.LoadWeights("WeightsSaves/WeightsRCT.fbp");
	auto out = Trainer.Run(&formatExpectedOutputAUDUSDCurrent, data);
	std::cout << "Total Error: " << out[1] << ", \tAverage Error: " << out[0] << ", \tHighest error: " << out[2] << ", \tLowest error: " << out[3] << std::endl;
	return Trainer;
}

vec<ddd> makeInput(vec<ddd> input)
{
	return input;
}
vec<ddd> makeOutput(vec<ddd> input)
{
	vec<ddd> output = {input[0], input[0], input[0]};
	return output;
}


NetworkTrainer TrainOffFunctionsTest(int inputs = 2, int data = 20000, int epochs = 250, vec<int> layers = {5, 7, 10, 13, 18, 26, 34, 21, 13, 7, 5, 3})
{
	int nextPo2 = (int)std::pow(2, std::ceil(std::log2(data)) + 1);
	NetworkTrainer Trainer = NetworkTrainer(inputs, layers);
	Trainer.TrainOffFunctions(&makeInput, &makeOutput, epochs, 0.00001, data);
	return Trainer;
}

// vec<vec<ddd>> optester() {
// 	size_t v1size = 1000;
// 	size_t v2size = 1000;
// 	vec<ddd> in(v1size);
// 	vec<ddd> in2(v2size);
// 	for (size_t i = 0; i < v1size; i++)
// 		in[i] = randomFunc();
// 	for (size_t i = 0; i < v2size; i++)
// 		in2[i] = randomFunc();
// 	auto out = optest(in, in2);
// 	vec<vec<ddd>> out2(v1size, vec<ddd>(v2size));
// 	for (size_t i = 0; i < out.size(); i++)
// 	{
// 		for (size_t j = 0; j < out[i].size(); j++)
// 		{
// 			out2[i][j] = in[i] * in2[j];
// 			if (out2[i][j] != out[i][j])
// 			{
// 				std::cerr << "Error at " << i << ", " << j << ": " << out2[i][j] << " != " << out[i][j] << std::endl;
// 				return {};
// 			}
// 		}
// 	}

// 	return out;
// }

int main()
{
	// TrainerTest();
	// smalltrainertest();
	//auto functionTrainer = TrainOffFunctionsTest();
	//optester();
	auto Trainer1 = CPULoadTest(5000, 25, {200, 100, 50, 20, 5}, 250);
	// auto Trainer2 = weightLoadCPUTest(1000);
	// largeTrainertestGPU();
	return 0;
}
