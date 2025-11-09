# FastBackpropagation Neural Network - AI Coding Agent Instructions

## Architecture Overview

This is a custom neural network implementation for financial time series prediction (AUDUSD forex data) with both CPU and experimental GPU acceleration via CUDA. The project follows a modular design with three core components:

- **`NeuralNetwork`**: Core network implementation with sigmoid activation, backpropagation learning, and weight persistence via `.fbp` binary format
- **`NetworkTrainer`**: High-level training interface supporting both CSV data files and function-based training
- **`Math.hpp/cpp`**: Custom linear algebra operations and forex-specific data formatters

## Key Development Patterns

### Build System
Always use the makefile targets, never compile manually:
- `make runtest` - Build with nvcc and run (primary workflow)
- `make debugbuild` - Build debug version with `-g` flag
- Architecture is hardcoded to `compute_86` in makefile

### Data Format Conventions
Financial data follows this specific pattern:
```cpp
// CSV: "date time \t open \t high \t low \t close \t volume"
// Parsed to: [high, low, close, volume] (open is dropped)
std::vector<std::vector<ddd>> formatAUDUSDData(std::ifstream*, int maxlines);
std::vector<ddd> formatExpectedOutputAUDUSDCurrent(std::vector<ddd> current, std::vector<ddd> next);
```

### Training Workflows
Two distinct training approaches exist:
1. **File-based**: `Trainer.Load()` → `Trainer.Train()` for CSV forex data
2. **Function-based**: `TrainOffFunctions()` for synthetic data (like XOR examples)

### Layer Architecture Patterns
Network topology follows specific conventions seen in `main.cpp`:
- Input count: 8 or 20 for forex, 2 for XOR
- Hidden layers: Start large, taper down (e.g., `{350, 200, 135, 90, 60, 20, 5}`)
- Final layer size matches expected output dimensionality

### Weight Persistence
Custom binary format with `.fbp` extension:
```cpp
// Save/load works at both NeuralNetwork and NetworkTrainer levels
trainer.SaveWeights("WeightsSaves/WeightsRCT.fbp");
trainer.LoadWeights("WeightsSaves/WeightsRCT.fbp");
```

### GPU vs CPU Patterns
GPU functions exist but are marked non-functional in comments:
- `RunGPU()`, `LearnGPU()`, `TrainOffFunctionsGPU()` are present but unreliable
- Always prefer CPU versions for working code
- GPU weight allocation via `AllocateGPUWeights()` when experimenting

## Critical Constants & Conventions

- Type alias: `ddd` = `double` (used throughout)
- Learning rates: 0.01-0.4 depending on problem complexity  
- Data sizing: Uses power-of-2 sizing (`nextPo2` calculations)
- Random initialization: `std::uniform_real_distribution<ddd>(-1, 1)`

## Testing & Validation Patterns

The `main.cpp` contains comprehensive test functions:
- `TrainOffFunctionsTest()` - XOR problem validation
- `CPULoadTest()` - Forex data training
- `weightLoadCPUTest()` - Model persistence validation

Always validate XOR learning (expect ~0 for XOR inputs, ~1 for OR inputs) before moving to complex forex training.

## File Organization Logic

- `incl/`: All header files (.hpp/.cuh)
- `src/`: Implementation files (.cpp/.cu)  
- `Data/Stock/AUDUSD/`: Forex CSV data
- `WeightsSaves/`: Trained model persistence
- `bin/`: Build outputs (created by make)

When adding features, maintain this strict separation and follow the existing include structure via `Project.hpp`.

## Network Structure

The network is structured as weights[i(layer, not including first layer)][j(neuron in layer)][k(0 = bias, 1 = weight from first neuron in previous layer to this neuron)].