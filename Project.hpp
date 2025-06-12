#pragma once
#ifndef PROJECT_HPP
# define PROJECT_HPP
# include <cuda.h>
# include <cuda_runtime.h>
# include <vector>
# include <memory>
# include <random>
# include <array>
# include <exception>
# include <iostream>
# include <fstream>
# include <chrono>
# include <thread>
# include <filesystem>
# include <cmath>

# define sptr std::shared_ptr
# define vec std::vector
# define ddd double
# include "CudaFuncs.h"
# include "incl/Math.hpp"
# include "incl/Acceleration.cuh"
# include "incl/NeuralNetwork.hpp"
# include "incl/NetworkTrainer.hpp"
#endif