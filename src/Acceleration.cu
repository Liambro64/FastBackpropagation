#include "../Project.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __global__ void outer_product(ddd *result, size_t resultPitch, ddd *v1, ddd *v2, size_t v1size, size_t v2size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < v1size && j < v2size)
	{
		result[i * resultPitch + j] = v1[i] * v2[j];
	}
}

extern "C" __global__ void NetworkSum(ddd *result, ddd *input, ddd *weights, size_t inputSize, size_t neurons)
{
	int neuron = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = 0; i < inputSize; i++)
	{
		if (neuron < neurons)
		{
			result[neuron] += input[i] * weights[neuron * inputSize + i + 1];
		}
	}
	result[neuron] += weights[neuron * inputSize]; // bias term
}

// should be a wrapper functuon to run multiple weighted sums in parallel
// extern "C" std::vector<ddd> weightedSumsWp(std::vector<ddd> outsideValues, std::vector<std::vector<ddd>> insideValues)
// {
// 	ddd *h_insides = new ddd [insideValues.size() * insideValues[0].size()];
// 	for (size_t i = 0; i < insideValues.size(); ++i)
// 	{
// 		for (size_t j = 0; j < insideValues[i].size(); ++j)
// 		{
// 			h_insides[i * insideValues[0].size() + j] = insideValues[i][j];
// 		}
// 	}
// 	ddd *d_outsideValues;
// 	ddd *d_insideValues;
// 	ddd *returnVals;
// 	size_t pitch;
// 	// allocate and copy inside values to device
// 	if (cudaMalloc(&d_outsideValues, outsideValues.size() * sizeof(ddd)) != CUDA_SUCCESS)
// 		std::cout << "Failed to alloc 1" << std::endl;

// 	if (cudaMemcpy(d_outsideValues, outsideValues.data(), outsideValues.size() * sizeof(ddd), cudaMemcpyHostToDevice) != CUDA_SUCCESS)
// 		std::cout << "Failed to copy 1" << std::endl;

// 	cudaMallocPitch(&d_insideValues, &pitch, sizeof(ddd) * insideValues[0].size(), insideValues.size());

// 	if (cudaMemcpy2D(d_insideValues, pitch, h_insides, insideValues[0].size() * sizeof(ddd),
// 					 insideValues[0].size() * sizeof(ddd), insideValues.size(), cudaMemcpyHostToDevice) != CUDA_SUCCESS)
// 		std::cout << "Failed to copy 2" << std::endl;
// 	if (cudaMalloc(&returnVals, insideValues.size() * sizeof(ddd)) != CUDA_SUCCESS)
// 		std::cout << "Failed to alloc 3" << std::endl;
// 	int isize = insideValues.size();
// 	int osize = outsideValues.size();
// 	int threadsPerBlock = 256;
// 	int blocksPerGrid = (insideValues.size() + threadsPerBlock - 1) / threadsPerBlock;
// 	dim3 blockSize = dim3(threadsPerBlock, 1, 1);
// 	dim3 gridSize = dim3(blocksPerGrid, 1, 1);
// 	weightedSumGPU<<<gridSize, blockSize>>>((ddd *)returnVals, (ddd *)d_outsideValues, d_insideValues, osize, isize, pitch / sizeof(ddd));

// 	cudaDeviceSynchronize();

// 	std::vector<ddd> result(insideValues.size());
// 	cudaMemcpy(result.data(), returnVals, insideValues.size() * sizeof(ddd), cudaMemcpyDeviceToHost);

// 	delete h_insides;
// 	return result;
// }
extern "C" std::pair<ddd *, std::vector<std::vector<ddd *>>> AllocateWeightsGPU(std::vector<std::vector<std::vector<ddd>>> weights)
{
	std::vector<std::vector<ddd *>> d_weights(weights.size());
	ddd *d_inputs;
	if (cudaMalloc(&d_inputs, (weights[0][0].size() - 1) * sizeof(ddd)) != cudaError_t::cudaSuccess)
	{
		std::cerr << "Failed to allocate device memory for inputs." << std::endl;
		std::__throw_bad_alloc();
	}
	size_t bpd = sizeof(ddd);
	for (int i = 0; i < weights.size(); i++)
	{
		for (int j = 0; j < weights[i].size(); j++)
		{
			ddd *d_weight;
			if (cudaMalloc(&d_weight, weights[i][j].size() * bpd) != cudaError_t::cudaSuccess)
			{
				std::cerr << "Failed to allocate device memory for weights." << std::endl;
				// free previously allocated memory
				for (int k = 0; k < i; k++)
				{
					for (int l = 0; l < d_weights[k].size(); l++)
					{
						cudaFree(d_weights[k][l]);
					}
				}
				for (int l = 0; l < j; l++)
				{
					cudaFree(d_weights[i][l]);
				}
				std::__throw_bad_alloc();
			}
			if (cudaMemcpy(d_weight, weights[i][j].data(), weights[i][j].size() * bpd, cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
			{
				std::cerr << "Failed to copy weights to device." << std::endl;
				cudaFree(d_weight);
				// free previously allocated memory
				for (int k = 0; k < i; k++)
				{
					for (int l = 0; l < d_weights[k].size(); l++)
					{
						cudaFree(d_weights[k][l]);
					}
				}
				for (int l = 0; l < j; l++)
				{
					cudaFree(d_weights[i][l]);
				}
				std::__throw_bad_alloc();
			}
			d_weights[i].push_back(d_weight);
		}
	}
	return std::pair<ddd *, std::vector<std::vector<ddd *>>>();
}

extern "C" std::vector<ddd> RunNetwork(std::vector<ddd> input, std::vector<std::vector<std::vector<ddd>>> weights)
{
	unsigned int bpd = sizeof(ddd); // bytes per ddd
	unsigned long inputSize = input.size();
	std::vector<int> allSizes;
	unsigned long totalSize = 0;
	unsigned long biggestLayer = inputSize;
	allSizes.resize(weights.size());
	for (int i = 0; i < weights.size(); i++)
	{
		allSizes[i] = weights[i][0].size();
		totalSize += allSizes[i];
		if (allSizes[i] > biggestLayer)
		{
			biggestLayer = allSizes[i];
		}
	}
	unsigned long bytesize = 0;
	for (int i = 0; i < allSizes.size(); i++)
	{
		bytesize += (allSizes[i] * ((i == 0 ? inputSize : allSizes[i - 1]) + 1));
	}
	bytesize *= bpd;
	ddd *host_weights = (ddd *)malloc(bytesize);
	ddd *host_input = (ddd *)malloc(inputSize * bpd);

	ddd *dev_weights;
	ddd *dev_input;
	ddd *dev_output;
	unsigned long count = 0;
	// flatten the weights.
	for (int i = 0; i < weights.size(); i++)
	{
		for (int j = 0; j < weights[i].size(); j++)
		{
			for (int k = 0; k < weights[i][j].size(); k++)
			{
				host_weights[count++] = weights[i][j][k];
			}
		}
	}
	// make sure input is usable with cuda
	for (int i = 0; i < inputSize; i++)
	{
		host_input[i] = input[i];
	}

	if (cudaMalloc(&dev_weights, bytesize) != cudaError_t::cudaSuccess)
	{
		std::cerr << "Failed to allocate device memory for weights." << std::endl;
		free(host_weights);
		free(host_input);
	}
	if (cudaMemcpy(dev_weights, host_weights, bytesize, cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		std::cerr << "Failed to copy weights to device." << std::endl;
		free(host_weights);
		free(host_input);
		cudaFree(dev_weights);
	}
	if (cudaMalloc(&dev_input, biggestLayer * bpd) != cudaError_t::cudaSuccess)
	{
		std::cerr << "Failed to allocate device memory for input." << std::endl;
		free(host_weights);
		free(host_input);
		cudaFree(dev_weights);
	}
	if (cudaMemcpy(dev_input, host_input, inputSize * bpd, cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		std::cerr << "Failed to copy input to device." << std::endl;
		free(host_weights);
		free(host_input);
		cudaFree(dev_weights);
		cudaFree(dev_input);
	}
	if (cudaMalloc(&dev_output, biggestLayer * bpd) != cudaError_t::cudaSuccess)
	{
		std::cerr << "Failed to allocate device memory for output." << std::endl;
		free(host_weights);
		free(host_input);
		cudaFree(dev_weights);
		cudaFree(dev_input);
	}
	size_t offset = 0;
	for (int i = 0; i < allSizes.size(); i++)
	{
		dim3 blockSize = dim3(64, 1, 1);
		dim3 gridSize = dim3((allSizes[i] + blockSize.x - 1) / blockSize.x, 1, 1);
		NetworkSum<<<gridSize, blockSize>>>(dev_output, dev_input, dev_weights + offset, i == 0 ? inputSize : allSizes[i - 1], allSizes[i]);
		cudaDeviceSynchronize();
		offset += (allSizes[i] * ((i == 0 ? inputSize : allSizes[i - 1]) + 1));
		if (i < allSizes.size() - 1)
		{
			if (cudaMemcpy(dev_input, dev_output, allSizes[i] * bpd, cudaMemcpyDeviceToDevice) != cudaError_t::cudaSuccess)
			{
				std::cerr << "Failed to copy output to input." << std::endl;
				free(host_weights);
				free(host_input);
				cudaFree(dev_weights);
				cudaFree(dev_input);
				cudaFree(dev_output);
			}
		}
	}
	ddd *host_output = (ddd *)malloc(allSizes[allSizes.size() - 1] * bpd);
	cudaMemcpy(host_output, dev_output, allSizes[allSizes.size() - 1] * bpd, cudaMemcpyDeviceToHost);
	std::vector<ddd> result;
	result.resize(allSizes[allSizes.size() - 1]);
	for (int i = 0; i < allSizes[allSizes.size() - 1]; i++)
	{
		result[i] = host_output[i];
	}
	free(host_weights);
	free(host_input);
	free(host_output);
	cudaFree(dev_weights);
	cudaFree(dev_input);
	cudaFree(dev_output);
	return result;
}

extern "C" std::vector<std::vector<ddd>> optest(std::vector<ddd> input, std::vector<ddd> input2)
{
	std::vector<std::vector<ddd>> result(input.size(), std::vector<ddd>(input2.size(), 0.0));
	ddd *h_v1 = input.data();
	ddd *h_v2 = input2.data();
	ddd *h_result;

	ddd *d_v1;
	ddd *d_v2;
	ddd *d_result_pitched;
	size_t result_pitch;
	size_t v1size = input.size();
	size_t v2size = input2.size();

	h_result = new ddd[v1size * v2size];

	if (cudaMalloc(&d_v1, v1size * sizeof(ddd)) != cudaError_t::cudaSuccess)
	{
		std::cerr << "Failed to allocate device memory for v1." << std::endl;
		return result;
	}
	if (cudaMemcpy(d_v1, h_v1, v1size * sizeof(ddd), cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		std::cerr << "Failed to copy v1 to device." << std::endl;
		cudaFree(d_v1);
		return result;
	}
	if (cudaMalloc(&d_v2, v2size * sizeof(ddd)) != cudaError_t::cudaSuccess)
	{
		std::cerr << "Failed to allocate device memory for v2." << std::endl;
		cudaFree(d_v1);
		return result;
	}
	if (cudaMemcpy(d_v2, h_v2, v2size * sizeof(ddd), cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		std::cerr << "Failed to copy v2 to device." << std::endl;
		cudaFree(d_v1);
		cudaFree(d_v2);
		return result;
	}
	if (cudaMallocPitch(&d_result_pitched, &result_pitch, v1size * sizeof(ddd), v2size) != cudaError_t::cudaSuccess)
	{
		std::cerr << "Failed to allocate device memory for result." << std::endl;
		cudaFree(d_v1);
		cudaFree(d_v2);
		return result;
	}
	dim3 blockSize = dim3(64, 64, 1);
	dim3 gridSize = dim3((v1size + blockSize.x - 1) / blockSize.x, (v2size + blockSize.y - 1) / blockSize.y, 1);
	outer_product<<<gridSize, blockSize>>>(d_result_pitched, result_pitch / sizeof(ddd), d_v1, d_v2, v1size, v2size);

	if (cudaDeviceSynchronize() != cudaError_t::cudaSuccess)
	{
		std::cerr << "Failed to synchronize device after kernel launch." << std::endl;
		cudaFree(d_v1);
		cudaFree(d_v2);
		cudaFree(d_result_pitched);
		return result;
	}

	if (cudaMemcpy2D(h_result, v1size * sizeof(ddd), d_result_pitched, result_pitch, v1size * sizeof(ddd), v2size, cudaMemcpyDeviceToHost) != cudaError_t::cudaSuccess)
	{
		std::cerr << "Failed to copy result back to host." << std::endl;
		cudaFree(d_v1);
		cudaFree(d_v2);
		cudaFree(d_result_pitched);
		return result;
	}

	for (size_t i = 0; i < v1size; i++)
	{
		for (size_t j = 0; j < v2size; j++)
		{
			result[i][j] = h_result[i * v2size + j];
		}
	}
	cudaFree(d_v1);
	cudaFree(d_v2);
	cudaFree(d_result_pitched);
	free(h_result);
	return result;
}

extern "C" int FullLearn(std::vector<ddd> input, std::vector<std::vector<std::vector<ddd>>> weights)
{
	// input is a std::vectortor of ddds
	return 0;
}
