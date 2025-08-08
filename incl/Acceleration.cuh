#pragma once
#ifndef ACCELERATION_CUH
# define ACCELERATION_CUH

# include "../Project.hpp"
extern "C" int FullRun(vec<ddd> input, vec<vec<vec<ddd>>> weights);

extern "C" vec<vec<ddd>> optest(vec<ddd> input, vec<ddd> input2);

#endif