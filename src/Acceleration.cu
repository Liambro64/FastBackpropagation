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
extern "C" __global__ void transpose(ddd *result, size_t resultPitch, ddd *input, size_t inputPitch, size_t width, size_t height)
{

	//result pitch A is the size of each layer
	//result pitch B is the size of each row
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < width && j < height)
	{
		result[i * resultPitch + j] = input[j * inputPitch + i];
	}
}
extern "C" __global__ void vector_matrix_multiply(ddd *result, ddd *vector, ddd *matrix, size_t matrixPitch, size_t vectorSize, size_t matrixCols)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < matrixCols)
	{
		ddd sum = 0.0;
		for (int i = 0; i < vectorSize; i++)
		{
			sum += vector[i] * matrix[i * matrixPitch + col];
		}
		result[col] = sum;
	}
}
extern "C" __global__ void untransposed_vector_matrix_multiply(ddd *result, ddd *vector, ddd *matrix, size_t matrix_x, size_t matrix_y) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < matrix_x) {
		for (int j = 0; j < matrix_y; j++)
		{
			atomicAdd(result + j, vector[i] * matrix[i * matrix_x + j + 1]);
		}
		
	}
}
extern "C" __global__ void vectorMult(ddd *vector, ddd *multiplierVector, size_t vectorSize) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < vectorSize)
		vector[i] *= multiplierVector[i] * (1 - multiplierVector[i]);
}
extern "C" __global__ void updateLayerWeights(ddd *weights, ddd *layer_outs, ddd *layer_errors, size_t layerSize, size_t prevLayerSize, ddd learningRate) {
	int neuron = blockIdx.x * blockDim.x + threadIdx.x;
	if (neuron < layerSize) {
		// Update bias weight
		weights[neuron * (prevLayerSize + 1)] += learningRate * layer_errors[neuron];
		//update weights
		for (int i = 0; i < prevLayerSize; i++) {
			weights[neuron * (prevLayerSize + 1) + i + 1] += learningRate * layer_errors[neuron] * layer_outs[i];
		}
	}
}
extern "C" __global__ void vectorSum(ddd *result, ddd *vector, size_t vectorSize) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < vectorSize) {
		atomicAdd(result, vector[i]);
	}
}
extern "C" __global__ void NetworkSum(ddd *result, ddd *input, ddd *weights, size_t inputSize, size_t neurons)
{
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron < neurons)
    {
        result[neuron] = weights[neuron * (inputSize + 1)];  // Initialize with bias (index 0)
        for (int i = 0; i < inputSize; i++)
        {
            result[neuron] += input[i] * weights[neuron * (inputSize + 1) + i + 1];  // Weights start at index 1
        }
        result[neuron] = 1.0 / (1.0 + exp(-result[neuron]));  // Sigmoid
    }
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


extern "C" std::vector<ddd *> AllocateWeightsGPU(std::vector<std::vector<std::vector<ddd>>> *weights)
{
    std::vector<ddd *> d_weights;
    size_t bpd = sizeof(ddd);
    for (int i = 0; i < weights->size(); i++)
    {
        size_t layer_size = (*weights)[i].size() * (*weights)[i][0].size();
        ddd *d_layer;
        if (cudaMalloc(&d_layer, layer_size * bpd) != cudaError_t::cudaSuccess)  
        {
            std::cerr << "Failed to allocate device memory for weights." << std::endl;
            // Free previously allocated memory
            for (int k = 0; k < i; k++)
            {
                cudaFree(d_weights[k]);
            }
            throw std::bad_alloc();  // Fix: Standard exception
        }
        size_t offset = 0;
        for (int j = 0; j < (*weights)[i].size(); j++)
        {
            cudaMemcpy(d_layer + offset, (*weights)[i][j].data(), (*weights)[i][j].size() * bpd, cudaMemcpyHostToDevice);
            offset += (*weights)[i][j].size();
        }
        d_weights.push_back(d_layer);
    }
    return d_weights;
}

void FreeWeightsGPU(std::vector<ddd *> &d_weights)
{
	for (int i = 0; i < d_weights.size(); i++)
	{
		cudaFree(d_weights[i]);
	}
	d_weights.clear();
}
std::vector<ddd> RunNetwork(std::vector<ddd> input, std::vector<ddd *> dev_weights, std::vector<int> allSizes)
{
    unsigned int bpd = sizeof(ddd);
    unsigned long inputSize = input.size();
    unsigned long biggestLayer = inputSize;
    for (int i = 0; i < dev_weights.size(); i++)
    {
        // Assuming you calculate allSizes elsewhere or pass it; for now, assume it's known
        // allSizes[i] = ...;  // You need to compute or pass layer sizes
        if (allSizes[i] > biggestLayer) biggestLayer = allSizes[i];
    }
    ddd *dev_input, *dev_output;
    if (cudaMalloc(&dev_input, biggestLayer * bpd) != cudaError_t::cudaSuccess) return {};
    if (cudaMemcpy(dev_input, input.data(), inputSize * bpd, cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
    {
        cudaFree(dev_input); return {};
    }
    if (cudaMalloc(&dev_output, biggestLayer * bpd) != cudaError_t::cudaSuccess)
    {
        cudaFree(dev_input); return {};
    }
    for (int i = 0; i < allSizes.size(); i++)
    {
        dim3 blockSize(64, 1, 1);
        dim3 gridSize((allSizes[i] + blockSize.x - 1) / blockSize.x, 1, 1);
        NetworkSum<<<gridSize, blockSize>>>(dev_output, dev_input, dev_weights[i], i == 0 ? inputSize : allSizes[i - 1], allSizes[i]);
        cudaDeviceSynchronize();  // Add error check
        if (i < allSizes.size() - 1)
        {
            cudaMemcpy(dev_input, dev_output, allSizes[i] * bpd, cudaMemcpyDeviceToDevice);
        }
    }
    std::vector<ddd> result(allSizes.back());

    cudaMemcpy(result.data(), dev_output, allSizes.back() * bpd, cudaMemcpyDeviceToHost);
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

extern "C" __global__ void InitialErrorCalculation(ddd *result, ddd *layer_outs, ddd *expected_values, size_t layerSize) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < layerSize) // V loss derivative				  V sigmoid derivative
		result[i] = (2 * (layer_outs[i] - expected_values[i])) * (layer_outs[i] * (1 - layer_outs[i]));
}

extern "C" ddd FullLearn(std::vector<ddd> input, std::vector<ddd> expected_values, std::vector<ddd *> dev_weights, std::vector<int> layerSizes, ddd alpha)
{
	// input is a std::vectortor of ddds
	std::vector<ddd *> outs; 	//will eventually contain all the outputs from each layer, including inputs.
	std::vector<ddd *> errors; 	//will eventually contain all the errors from each layer, excluding inputs.
	int inputSize = input.size();
	int expectedSize = expected_values.size();
	ddd *dev_inputs;
	if (cudaMalloc(&dev_inputs, sizeof(ddd) * inputSize) != cudaError_t::cudaSuccess)
	{
		return -1;
	}
	cudaMemcpy(dev_inputs, input.data(), sizeof(ddd) * inputSize, cudaMemcpyHostToDevice);
	ddd *dev_expected;
	cudaMalloc(&dev_expected, sizeof(ddd) * expectedSize); //allocate memory for the expected values on the gpu
	cudaMemcpy(dev_expected, expected_values.data(), sizeof(ddd) * expectedSize, cudaMemcpyHostToDevice);
	outs.push_back(dev_inputs);
	for (int i = 0; i < layerSizes.size(); i++) {
		ddd *dev_layer_out;
		ddd *dev_layer_error;
		cudaMalloc(&dev_layer_out, sizeof(ddd) * layerSizes[i]);
		cudaMalloc(&dev_layer_error, sizeof(ddd) * layerSizes[i]);
		cudaMemset(&dev_layer_out, 0, sizeof(ddd) * layerSizes[i]);
		cudaMemset(&dev_layer_error, 0, sizeof(ddd) * layerSizes[i]);
		outs.push_back(dev_layer_out);
		errors.push_back(dev_layer_error);
	}
	for (int i = 0; i < dev_weights.size(); i++) {
		int prev_layer_size = i == 0 ? inputSize : layerSizes[i - 1];
		int curr_layer_size = layerSizes[i];
		dim3 blockSize(64, 1, 1);
        	dim3 gridSize((layerSizes[i] + blockSize.x - 1) / blockSize.x, 1, 1);
        	NetworkSum<<<gridSize, blockSize>>>(outs[i + 1], outs[i], dev_weights[i], prev_layer_size, curr_layer_size);
		cudaDeviceSynchronize();
	}
	dim3 IECblockSize(64, 1, 1);
	dim3 IECgridSize((expectedSize + IECblockSize.x - 1) / IECblockSize.x, 1, 1);
	InitialErrorCalculation<<<IECgridSize, IECblockSize>>>(errors[errors.size() - 1], outs[outs.size() - 1], dev_expected, expectedSize);
	cudaDeviceSynchronize();
	for (int i = dev_weights.size() - 1; i > 0; i--) {
		int prev_layer_size = i == 0 ? inputSize : layerSizes[i - 1];
		int curr_layer_size = layerSizes[i];
		dim3 UTPVMMblockSize(64, 1, 1);
		dim3 UTPVMMgridSize((layerSizes[i] + UTPVMMblockSize.x - 1) / UTPVMMblockSize.x, 1, 1);
		untransposed_vector_matrix_multiply<<<UTPVMMgridSize, UTPVMMblockSize>>>(errors[i - 1], errors[i], dev_weights[i], curr_layer_size, prev_layer_size);
        cudaDeviceSynchronize();
		dim3 VMblockSize(64, 1, 1);
		dim3 VMgridSize((layerSizes[i - 1] + VMblockSize.x - 1) / VMblockSize.x, 1, 1);
		vectorMult<<<VMgridSize, VMblockSize>>>(errors[i - 1], outs[i], layerSizes[i - 1]);
		cudaDeviceSynchronize();
	}
	//from here, update the weights based on the errors (and outputs for the weights)
	for (int i = 0; i < dev_weights.size(); i++) {
		dim3 ULWblockSize(64, 1, 1);
		dim3 ULWgridSize((layerSizes[i] + ULWblockSize.x - 1) / ULWblockSize.x, 1, 1);
		updateLayerWeights<<<ULWgridSize, ULWblockSize>>>(dev_weights[i], outs[i], errors[i], layerSizes[i], i == 0 ? inputSize : layerSizes[i - 1], alpha);
		cudaDeviceSynchronize();
	}
	ddd *dev_final_error;
	cudaMalloc(&dev_final_error, sizeof(ddd));
	dim3 vectorSumBlockSize(64, 1, 1);
	dim3 vectorSumGridSize((layerSizes[layerSizes.size() - 1] + vectorSumBlockSize.x - 1) / vectorSumBlockSize.x, 1, 1);
	vectorSum<<<vectorSumGridSize, vectorSumBlockSize>>>(dev_final_error, errors[errors.size() - 1], layerSizes[layerSizes.size() - 1]);
	ddd final_error;
	cudaMemcpy(&final_error, dev_final_error, sizeof(ddd), cudaMemcpyDeviceToHost);
	cudaFree(dev_final_error);
	final_error /= (ddd)expectedSize;
	//std::cout << "Final error: " << final_error << std::endl;
	cudaDeviceSynchronize();
	for (int i = 0; i < outs.size(); i++) {
		cudaFree(outs[i]);
	}
	for (int i = 0; i < errors.size(); i++) {
		cudaFree(errors[i]);
	}
	cudaFree(dev_expected);
	return final_error;
}
