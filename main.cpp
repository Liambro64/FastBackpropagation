#include "Project.hpp"

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<ddd> dis(-1, 1);
double randomFunc() {
	return dis(gen);
}
void NetworkTest() {
	vec<int> layers = {4, 20, 35, 20, 12, 1};
	NeuralNetwork network = NeuralNetwork(layers, randomFunc);
	vec<ddd> ins(4);
	for (int i = 0; i < 4; i++)
		ins[i] = randomFunc();
	network.Run(&ins);
}

void TrainerTest() {
	vec<int> layers = {4, 20, 35, 20, 12, 1};
	NetworkTrainer Trainer = NetworkTrainer(layers);
	Trainer.Load("Data/Stock/AUDUSD/Data15M.csv");
	Trainer.Train(1000, 0.01, 1000);
}

int main()
{
	TrainerTest();
	return 0;
}
