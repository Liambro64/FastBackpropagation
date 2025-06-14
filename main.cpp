#include "Project.hpp"

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<ddd> dis(-1, 1);
double randomFunc() {
	return dis(gen);
}
void NetworkTest() {
	vec<int> layers = {400, 250, 120, 40, 12, 6};
	NeuralNetwork network = NeuralNetwork(8, layers, randomFunc);
	vec<ddd> ins(4);
	for (int i = 0; i < 4; i++)
		ins[i] = randomFunc();
	network.Run(&ins);
}

void TrainerTest() {
	vec<int> layers = {500, 350, 200, 120, 60, 20, 6};
	NetworkTrainer Trainer = NetworkTrainer(8, layers);
	Trainer.Load("Data/Stock/AUDUSD/Data15M.csv", &formatAUDUSDData, 25600);
	Trainer.Train(&formatExpectedOutputAUDUSDCurrent, 50, 0.05, 25000, 25);
}
void smalltrainertest() {
	vec<int> layers = {350, 200, 135, 90, 60, 20, 6};
	int data = 5000;
	int epochs = 400;
	int nextPo2 = (int)std::pow(2, std::ceil(std::log2(data)));
	NetworkTrainer Trainer = NetworkTrainer(8, layers);
	Trainer.Load("Data/Stock/AUDUSD/Data15M.csv", &formatAUDUSDData, nextPo2);
	Trainer.Train(&formatExpectedOutputAUDUSDCurrent, 400, 0.4, 5000, 50);
}

//not working
void largeTrainertestGPU() {
	vec<int> layers = {50, 125, 200, 345, 500, 415, 255, 135, 90, 60, 20, 1};
	NetworkTrainer Trainer = NetworkTrainer(8, layers);
	Trainer.Load("Data/Stock/AUDUSD/Data15M.csv", &formatAUDUSDData, 25600);

	Trainer.TrainGPU(10, 0.4, 25000);
}

void largedatatrainertest() {
	vec<int> layers = {90, 25, 6};
	int data = 50000;
	int epochs = 400;
	int nextPo2 = (int)std::pow(2, std::ceil(std::log2(data)));
	NetworkTrainer Trainer = NetworkTrainer(8, layers);
	Trainer.Load("Data/Stock/AUDUSD/Data15M.csv", &formatAUDUSDData, nextPo2);
	Trainer.Train(&formatExpectedOutputAUDUSDCurrent, 400, 0.4, data, 500);
}
void realCPUTest() {
	vec<int> layers = {350, 200, 135, 90, 60, 20, 5};
	int data = 100000;
	int epochs = 50;
	int inputs = 20;
	int nextPo2 = (int)std::pow(2, std::ceil(std::log2(data)));
	NetworkTrainer Trainer = NetworkTrainer(inputs, layers);
	Trainer.Load("Data/Stock/AUDUSD/Data15M.csv", &formatAUDUSDData, nextPo2);
	Trainer.Train(&formatExpectedOutputAUDUSDCurrent, epochs, 0.02, data, 25);
	Trainer.SaveWeights("WeightsSaves/WeightsRCT.fbp");
}
int main()
{
	//TrainerTest();
	//smalltrainertest();
	realCPUTest();
	//largeTrainertestGPU();
	return 0;
}
