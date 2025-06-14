#include "../Project.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __global__ void outerProductGPU(double **to, double *a, double *b, int asize, int bsize)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < asize)
		for (int j = 0; j < bsize; ++j)
			to[idx][j] = a[idx] * b[j];
}

extern "C" __global__ void vectorMatrixMultiplyGPU(ddd *to, ddd *vector, ddd *matrix, int vectorSize, int matrixRows, int numpitch)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < vectorSize)
	{
		matrix += idx; // Adjust pointer to the correct column
		ddd sum = 0.0;
		for (int i = 0; i < matrixRows; i++)
		{
			sum += vector[idx] * matrix[i * numpitch];
		}
		to[idx] = sum;
	}
}

extern "C" __global__ void weightedSumGPU(ddd *to, ddd *outsideValues, ddd *insideValues, int outsideSize, int insideSize, int numpitch)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < insideSize)
	{
		insideValues += idx * numpitch; // Adjust pointer to the correct row
		ddd sum = insideValues[0];		// Start with the bias
		for (int i = 0; i < outsideSize; i++)
		{
			sum += outsideValues[i] * insideValues[i + 1];
		}
		to[idx] = 1 / (1 + exp(-sum));
	}
}

extern "C" __global__ void weightedSumGPUInside(ddd *to, ddd *outsideValues, ddd **ptrs, int ptrIndex, int outsideSize, int insideSize, int numpitch)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	double *insideValues = ptrs[ptrIndex]; // Get the pointer to the inside values for this index
	if (idx < insideSize)
	{
		insideValues += idx * numpitch; // Adjust pointer to the correct row
		ddd sum = insideValues[0];		// Start with the bias
		for (int i = 0; i < outsideSize; i++)
		{
			sum += outsideValues[i] * insideValues[i + 1];
		}
		to[idx] = 1 / (1 + exp(-sum));
	}
}
// extern "C" __global__ void vectorSum(int *to, int **from, int size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx < size) {
//         int *current_from_array = from[idx];

//         // Ensure the pointer is not null before dereferencing
//         if (current_from_array != nullptr) {
//             int num_elements_to_sum = current_from_array[0];
//             int current_sum = 0;

//             // Sum elements from current_from_array[1] to current_from_array[num_elements_to_sum]
//             for (int j = 1; j <= num_elements_to_sum; j++) {
//                 current_sum += current_from_array[j];
//             }
//             to[idx] = current_sum;
//         }
//     }
// }

// namespace Wrapper{

// void VSWrapper(int *to, int **from, int size) {
//     // Define the number of threads and blocks
//     int threadsPerBlock = 256;
//     int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

//     // Launch the kernel
//     vectorSum<<<blocksPerGrid, threadsPerBlock>>>(to, from, size);
//     // Check for errors in kernel launch (optional)
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
//     }
// 	err = cudaDeviceSynchronize(); // Wait for kernel to complete
//     if (err != cudaSuccess) {
//         fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
//     }
// }
// }

// should be a wrapper functuon to run multiple weighted sums in parallel
extern "C" vec<ddd> weightedSumsWp(vec<ddd> outsideValues, vec<vec<ddd>> insideValues)
{
	double *h_insides = new double [insideValues.size() * insideValues[0].size()];
	for (size_t i = 0; i < insideValues.size(); ++i)
	{
		for (size_t j = 0; j < insideValues[i].size(); ++j)
		{
			h_insides[i * insideValues[0].size() + j] = insideValues[i][j];
		}
	}
	double *d_outsideValues;
	double *d_insideValues;
	double *returnVals;
	size_t pitch;
	// allocate and copy inside values to device
	if (cudaMalloc(&d_outsideValues, outsideValues.size() * sizeof(double)) != CUDA_SUCCESS)
		std::cout << "Failed to alloc 1" << std::endl;

	if (cudaMemcpy(d_outsideValues, outsideValues.data(), outsideValues.size() * sizeof(double), cudaMemcpyHostToDevice) != CUDA_SUCCESS)
		std::cout << "Failed to copy 1" << std::endl;
	if (cudaMallocPitch(&d_insideValues, &pitch, sizeof(double) * insideValues[0].size(), insideValues.size()) != CUDA_SUCCESS)
		std::cout << "Failed to alloc 2 (pitch)" << std::endl;
	if (cudaMemcpy2D(d_insideValues, pitch, h_insides, insideValues[0].size() * sizeof(ddd),
					 insideValues[0].size() * sizeof(ddd), insideValues.size(), cudaMemcpyHostToDevice) != CUDA_SUCCESS)
		std::cout << "Failed to copy 2" << std::endl;
	if (cudaMalloc(&returnVals, insideValues.size() * sizeof(double)) != CUDA_SUCCESS)
		std::cout << "Failed to alloc 3" << std::endl;
	int isize = insideValues.size();
	int osize = outsideValues.size();
	int threadsPerBlock = 256;
	int blocksPerGrid = (insideValues.size() + threadsPerBlock - 1) / threadsPerBlock;
	dim3 blockSize = dim3(threadsPerBlock, 1, 1);
	dim3 gridSize = dim3(blocksPerGrid, 1, 1);
	weightedSumGPU<<<gridSize, blockSize>>>((double *)returnVals, (double *)d_outsideValues, d_insideValues, osize, isize, pitch / sizeof(double));

	cudaDeviceSynchronize();

	vec<ddd> result(insideValues.size());
	cudaMemcpy(result.data(), returnVals, insideValues.size() * sizeof(ddd), cudaMemcpyDeviceToHost);

	delete h_insides;
	return result;
}

extern "C" vec<vec<ddd>> FullRun(vec<ddd> input, vec<vec<vec<ddd>>> weights) {
	// input is a vector of doubles
	// weights is a vector of matrices, each matrix is a vector of vectors of doubles
	size_t size = 0;
	vec<size_t> sizes(weights.size());
	for (int i = 0; i < weights.size(); i++) {
		sizes[i] = weights[i].size() * weights[i][0].size();
		size += sizes[i] * sizeof(double);
	}
	double *h_flatweights = new double[size];
	int i = 0;
	for (int j = 0; j < weights.size(); j++) {
		for (int k = 0; k < weights[j].size(); k++) {
			for (int l = 0; l < weights[j][k].size(); l++) {
				h_flatweights[i++] = weights[j][k][l];
			}
		}
	}
	// this is an array (on the host) of pointers (on the device) to the weights
	double **d_weights = new double *[weights.size()];
	//this is for the array of pointers after its copied on the device
	double **d_weights_pointers;
	//the inputs on the device
	double *d_input;
	//the return value (on the device) but also reused as the input layer
	double *d_returnVals;

	size_t maxInputSize = MAX(max(sizes), input.size());

	size_t *pitches = new size_t[weights.size()];
	for (int i = 0; i < weights.size(); i++) {
		cudaMallocPitch(&(d_weights[i]), &(pitches[i]), weights[i][0].size() * sizeof(double), weights[i].size());
		cudaMemcpy2D(d_weights[i], pitches[i], h_flatweights + sumFor(sizes, i), weights[i][0].size() * sizeof(double),
				 weights[i][0].size() * sizeof(double), weights[i].size(), cudaMemcpyHostToDevice);
	}
	cudaMalloc(&(d_weights_pointers), weights.size() * sizeof(double *));
	cudaMemcpy(d_weights_pointers, d_weights, weights.size() * sizeof(double *), cudaMemcpyHostToDevice);
	cudaMalloc(&d_input, maxInputSize * sizeof(double));
	cudaMemcpy(d_input, input.data(), input.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc(&d_returnVals, maxInputSize * sizeof(double));
	int threadsPerBlock = 256;
	vec<vec<ddd>> returnVals(weights.size());
	for (int i = 0; i < weights.size(); i++) {
		returnVals[i].resize(weights[i].size());
		dim3 block = dim3(threadsPerBlock, 1, 1);
		dim3 grid = dim3(((i == 0 ? input.size() : weights[i - 1].size()) + threadsPerBlock - 1) / threadsPerBlock, 1, 1);
		weightedSumGPUInside<<<block, grid>>>(d_returnVals, d_input, d_weights_pointers, i, i == 0 ? input.size() : sizes[i - 1], weights[i].size(), pitches[i] / sizeof(double));
		cudaDeviceSynchronize();
		cudaMemcpy(returnVals[i].data(), d_returnVals, weights[i].size() * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(d_input, d_returnVals, weights[i].size() * sizeof(double), cudaMemcpyDeviceToDevice);
	}

	return returnVals;
}
