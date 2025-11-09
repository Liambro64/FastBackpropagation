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
	
	std::vector<int> layers = {400, 250, 120, 40, 12, 6};
	NeuralNetwork network = NeuralNetwork(8, layers, randomFunc);
	std::vector<ddd> ins(4);
	for (int i = 0; i < 4; i++)
		ins[i] = randomFunc();
	network.Run(&ins);
}

void TrainerTest()
{
	std::vector<int> layers = {500, 350, 200, 120, 60, 20, 6};
	NetworkTrainer Trainer = NetworkTrainer(8, layers);
	Trainer.Load("Data/Stock/AUDUSD/Data15M.csv", &formatAUDUSDData, 25600);
	Trainer.Train(&formatExpectedOutputAUDUSDCurrent, 50, 0.05, 25000, 25);
}
void smalltrainertest()
{
	std::vector<int> layers = {350, 200, 135, 90, 60, 20, 6};
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
// 	std::vector<int> layers = {50, 125, 200, 345, 500, 415, 255, 135, 90, 60, 20, 1};
// 	NetworkTrainer Trainer = NetworkTrainer(8, layers);
// 	Trainer.Load("Data/Stock/AUDUSD/Data15M.csv", &formatAUDUSDData, 25600);

// 	Trainer.TrainGPU(10, 0.4, 25000);
// }

void largedatatrainertest()
{
	std::vector<int> layers = {90, 25, 6};
	int data = 50000;
	int epochs = 400;
	int nextPo2 = (int)std::pow(2, std::ceil(std::log2(data)));
	NetworkTrainer Trainer = NetworkTrainer(8, layers);
	Trainer.Load("Data/Stock/AUDUSD/Data15M.csv", &formatAUDUSDData, nextPo2);
	Trainer.Train(&formatExpectedOutputAUDUSDCurrent, 400, 0.4, data, 500);
}
NetworkTrainer saveCPUtest(int data = 50000, int epochs = 20)
{
	std::vector<int> layers = {350, 200, 135, 90, 60, 20, 5};
	int inputs = 20;
	int nextPo2 = (int)std::pow(2, std::ceil(std::log2(data)) + 1);
	NetworkTrainer Trainer = NetworkTrainer(inputs, layers);
	Trainer.Load("Data/Stock/AUDUSD/Data15M.csv", &formatAUDUSDData, nextPo2);
	Trainer.Train(&formatExpectedOutputAUDUSDCurrent, epochs, 0.02, data, 25);
	Trainer.SaveWeights("WeightsSaves/WeightsRCT.fbp");
	return Trainer;
}
NetworkTrainer CPULoadTest(int data = 50000, int epochs = 25, std::vector<int> layers = {500, 200, 70, 30, 5}, int printafter = 25)
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
	std::vector<int> layers = {350, 200, 135, 90, 60, 20, 5};
	int inputs = 20;
	int nextPo2 = (int)std::pow(2, std::ceil(std::log2(data)) + 1);
	NetworkTrainer Trainer = NetworkTrainer(inputs, layers);
	Trainer.Load("Data/Stock/AUDUSD/Data15M.csv", &formatAUDUSDData, nextPo2);
	Trainer.LoadWeights("WeightsSaves/WeightsRCT.fbp");
	auto out = Trainer.Run(&formatExpectedOutputAUDUSDCurrent, data);
	std::cout << "Total Error: " << out[1] << ", \tAverage Error: " << out[0] << ", \tHighest error: " << out[2] << ", \tLowest error: " << out[3] << std::endl;
	return Trainer;
}


std::vector<ddd> makeXorInput(int input)
{
	return input == 0 ? std::vector<ddd>{0, 1} : input == 1 ? std::vector<ddd>{1, 0} : input == 2 ? std::vector<ddd>{1, 1} : std::vector<ddd>{0, 0};
}
std::vector<ddd> makeXorOutput(int input)
{
	return input == 0 ? std::vector<ddd>{1} : input == 1 ? std::vector<ddd>{1} : input == 2 ? std::vector<ddd>{0} : std::vector<ddd>{0};
}

std::vector<ddd> makeInput(std::vector<ddd> input)
{
	static unsigned long long counter = 0;
	return makeXorInput(counter++ % 4);
}
std::vector<ddd> makeOutput(std::vector<ddd> input)
{
	static unsigned long long counter = 0;
	return makeXorOutput(counter++ % 4);
}
NetworkTrainer TrainOffFunctionsTest(int inputs = 2, int data = 20000, int epochs = 250, std::vector<int> layers = {5, 7, 10, 13, 18, 26, 34, 21, 13, 7, 5, 3})
{
	int nextPo2 = (int)std::pow(2, std::ceil(std::log2(data)) + 1);
	NetworkTrainer Trainer = NetworkTrainer(inputs, layers);
	Trainer.TrainOffFunctions(&makeInput, &makeOutput, epochs, 0.1, data);
	return Trainer;
}
NetworkTrainer TrainOffFunctionsTestGPU(int inputs = 2, int data = 20000, int epochs = 250, std::vector<int> layers = {5, 7, 10, 13, 18, 26, 34, 21, 13, 7, 5, 3})
{
	int nextPo2 = (int)std::pow(2, std::ceil(std::log2(data)) + 1);
	NetworkTrainer Trainer = NetworkTrainer(inputs, layers);
	Trainer.TrainOffFunctionsGPU(&makeInput, &makeOutput, epochs, 0.1, data);
	return Trainer;
}

// std::vector<std::vector<ddd>> optester() {
// 	size_t v1size = 1000;
// 	size_t v2size = 1000;
// 	std::vector<ddd> in(v1size);
// 	std::vector<ddd> in2(v2size);
// 	for (size_t i = 0; i < v1size; i++)
// 		in[i] = randomFunc();
// 	for (size_t i = 0; i < v2size; i++)
// 		in2[i] = randomFunc();
// 	auto out = optest(in, in2);
// 	std::vector<std::vector<ddd>> out2(v1size, std::vector<ddd>(v2size));
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
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	auto Trainer1 = TrainOffFunctionsTest(2, 5000, 50, {7, 1});
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Training Time (CPU): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;
	Trainer1.SaveWeights("WeightsSaves/WeightsXORCPU.fbp");
	begin = std::chrono::steady_clock::now();
	auto Trainer2 = TrainOffFunctionsTestGPU(2, 5000, 50, {7, 1});
	end = std::chrono::steady_clock::now();
	std::cout << "Training Time (GPU): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;
	auto testInput = std::vector<std::vector<ddd>>{std::vector<ddd>{1, 0}, std::vector<ddd>{0, 1}, std::vector<ddd>{1, 1}, std::vector<ddd>{0, 0}};
	//NeuralNetwork nn = Trainer1.getNetwork();
	//nn.AllocateGPUWeights();
	//for (int i = 0; i < testInput.size(); i++)
	//{
	//	auto out = nn.Run(&(testInput[i]));
	//	std::cout << "Input: [" << testInput[i][0] << ", " << testInput[i][1] << "] \tOutput: " << out[0] << std::endl;
	//	out = nn.RunGPU(&(testInput[i]));
	//	std::cout << "Input: [" << testInput[i][0] << ", " << testInput[i][1] << "] \tOutput(GPU): " << out[0] << std::endl;
	//	out = nn.RunGPU(&(testInput[i]));
	//	std::cout << "Input: [" << testInput[i][0] << ", " << testInput[i][1] << "] \tOutput(GPU2): " << out[0] << std::endl;
	//}
	// auto Trainer2 = weightLoadCPUTest(1000);
	// largeTrainertestGPU();
	return 0;
}
