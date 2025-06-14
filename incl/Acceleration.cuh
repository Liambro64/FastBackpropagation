#pragma once
#ifndef ACCELERATION_CUH
# define ACCELERATION_CUH

# include "../Project.hpp"
extern "C" vec<ddd> weightedSumsWp(vec<ddd> outsideValues, vec<vec<ddd>> insideValues);
extern "C" vec<vec<ddd>> FullRun(vec<ddd> input, vec<vec<vec<ddd>>> weights);

#endif