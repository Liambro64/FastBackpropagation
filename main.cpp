#include "Project.hpp"

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<ddd> dis(-1, 1);
double randomFunc() {
	return dis(gen);
}
void NetworkTest() {
	vec<int> layers = {4, 100, 200, 250, 120, 40, 12, 1};
	NeuralNetwork network = NeuralNetwork(8, layers, randomFunc);
	vec<ddd> ins(4);
	for (int i = 0; i < 4; i++)
		ins[i] = randomFunc();
	network.Run(&ins);
}

void TrainerTest() {
	vec<int> layers = {100, 200, 350, 200, 120, 60, 20, 6, 1};
	NetworkTrainer Trainer = NetworkTrainer(8, layers);
	Trainer.Load("Data/Stock/AUDUSD/Data15M.csv", 25600);
	Trainer.Train(50, 0.05, 25000, 25);
}
void smalltrainertest() {
	vec<int> layers = {50, 125, 200, 135, 90, 60, 20, 2};
	int data = 5000;
	int epochs = 400;
	int nextPo2 = (int)std::pow(2, std::ceil(std::log2(data)));
	NetworkTrainer Trainer = NetworkTrainer(8, layers);
	Trainer.Load("Data/Stock/AUDUSD/Data15M.csv", nextPo2);
	Trainer.Train(400, 0.4, 5000, 50);
}
void largeTrainertestGPU() {
	vec<int> layers = {50, 125, 200, 345, 500, 415, 255, 135, 90, 60, 20, 1};
	NetworkTrainer Trainer = NetworkTrainer(8, layers);
	Trainer.Load("Data/Stock/AUDUSD/Data15M.csv", 25600);

	Trainer.TrainGPU(10, 0.4, 2500);
}

void largedatatrainertest() {
	vec<int> layers = {90, 25, 2};
	int data = 50000;
	int epochs = 400;
	int nextPo2 = (int)std::pow(2, std::ceil(std::log2(data)));
	NetworkTrainer Trainer = NetworkTrainer(8, layers);
	Trainer.Load("Data/Stock/AUDUSD/Data15M.csv", nextPo2);
	Trainer.Train(400, 0.4, data, 500);
}
int main()
{
	//TrainerTest();
	//smalltrainertest();
	//largedatatrainertest();
	NetworkTrainer trainer = NetworkTrainer(8, {680, 500, 330, 200, 100, 5, 2});
	vec<ddd> input = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
	trainer.RunGPU(&input);
	//largeTrainertestGPU();
	return 0;
}
