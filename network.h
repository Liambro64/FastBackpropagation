#pragma once
#ifndef C_NEURAL_NETWORK
#define C_NEURAL_NETWORK

#include "project.h"

double ***CreateNeuralNetwork(int inputs, int *layers, int layersAmount);
void destroyNetwork(double ***network, int inputs, int *layers, int layersAmount);
double *Run(double ***network, double *inputs, int inputSize, int *layers, int layersAmount);
double Learn(double ***network, double *inputs, double *expected_out, int inputSize, int *layerSizes, int layersAmount);

#endif
