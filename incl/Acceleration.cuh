#pragma once
#ifndef ACCELERATION_CUH
# define ACCELERATION_CUH

# include "../Project.hpp"
extern "C" int FullRun(std::vector<ddd> input, std::vector<std::vector<std::vector<ddd>>> weights);

extern "C" std::vector<std::vector<ddd>> optest(std::vector<ddd> input, std::vector<ddd> input2);

#endif