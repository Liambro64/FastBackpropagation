#ifndef MATH_HPP
# define MATH_HPP

# include "../Project.hpp"

ddd 				LossFunction(ddd prediction, ddd expected, int size);
ddd					LossFunction(vec<ddd> prediction, vec<ddd> expected);
ddd 				LossDerivative(ddd prediction, ddd expected);
ddd 				sigmoid(ddd x);
ddd 				sigmoidDerivative(ddd x);
vec<ddd> 			add(const vec<ddd>& v1, const vec<ddd>& v2);
vec<ddd>			subtract(const vec<ddd>& v1, const vec<ddd>& v2);
ddd 				dot_product(const vec<ddd>& v1, const vec<ddd>& v2) ;
vec<vec<ddd>>		add(const vec<vec<ddd>>& m1, const vec<vec<ddd>>& m2);
vec<vec<ddd>>		subtract(const vec<vec<ddd>>& m1, const vec<vec<ddd>>& m2);
vec<vec<ddd>>		outerProduct(const vec<ddd> &a, const vec<ddd> &b);
vec<ddd>			scalar_multiply(ddd scalar, const vec<ddd>& vector);
vec<vec<ddd>>		scalar_multiply(ddd scalar, const vec<vec<ddd>>& matrix);
vec<ddd>			matrix_vector_multiply(const vec<vec<ddd>>& matrix, const vec<ddd>& vector);
vec<ddd>			vector_matrix_multiply(const vec<ddd>& vector, const vec<vec<ddd>>& matrix);
vec<vec<ddd>>		transpose(const vec<vec<ddd>>& matrix);
vec<vec<ddd>>		extractBiases(const vec<vec<vec<ddd>>>& weights);
vec<vec<vec<ddd>>>	extractWeights(const vec<vec<vec<ddd>>>& weights);
void				InjectBiases(vec<vec<vec<ddd>>>& weights, const vec<vec<ddd>>& extractedBiases);
void				InjectWeights(vec<vec<vec<ddd>>>& weights, const vec<vec<vec<ddd>>>& extractedWeights);
ddd 				weightedSum(vec<ddd> outsideValues, vec<ddd> insideValues);
vec<ddd> 			weightedSums(vec<ddd> outsideValues, vec<vec<ddd>> insideValues);
vec<vec<ddd>> 		SumAll(vec<ddd> input, vec<vec<vec<ddd>>> weights);
vec<ddd>			NetworkRunSum(vec<ddd> input, vec<vec<vec<ddd>>> weights);
void				Copy(vec<ddd> *to, const vec<ddd> &from);
vec<vec<vec<ddd>>>	transpose(const vec<vec<vec<ddd>>>& matrix);
std::string			milisToString(int64_t milliseconds);
std::string			milisToString(double milliseconds);
size_t				sumFor(vec<size_t> &v, size_t max);
size_t				max(vec<size_t> &v);
#endif