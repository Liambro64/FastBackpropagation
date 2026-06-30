#ifndef MATH_HPP
# define MATH_HPP

# include "../Project.hpp"

ddd activationFunction(ddd x);
ddd activationFunctionDerivative(ddd x);
ddd 				LossFunction(ddd prediction, ddd expected, int size);
ddd					LossFunction(std::vector<ddd> prediction, std::vector<ddd> expected);
ddd 				LossDerivative(ddd prediction, ddd expected);
ddd 				sigmoid(ddd x);
ddd 				sigmoidDerivative(ddd x);
std::vector<std::vector<ddd>>		formatAUDUSDData(std::ifstream *file, int maxlines = -1);
std::vector<ddd> 			add(const std::vector<ddd>& v1, const std::vector<ddd>& v2);
std::vector<ddd>			subtract(const std::vector<ddd>& v1, const std::vector<ddd>& v2);
ddd 				dot_product(const std::vector<ddd>& v1, const std::vector<ddd>& v2) ;
std::vector<std::vector<ddd>>		add(const std::vector<std::vector<ddd>>& m1, const std::vector<std::vector<ddd>>& m2);
std::vector<std::vector<ddd>>		subtract(const std::vector<std::vector<ddd>>& m1, const std::vector<std::vector<ddd>>& m2);
std::vector<std::vector<ddd>>		outerProduct(const std::vector<ddd> &a, const std::vector<ddd> &b);
std::vector<ddd>			scalar_multiply(ddd scalar, const std::vector<ddd>& vector);
std::vector<std::vector<ddd>>		scalar_multiply(ddd scalar, const std::vector<std::vector<ddd>>& matrix);
std::vector<ddd>			matrix_vector_multiply(const std::vector<std::vector<ddd>>& matrix, const std::vector<ddd>& vector);
std::vector<ddd>			vector_matrix_multiply(const std::vector<ddd>& vector, const std::vector<std::vector<ddd>>& matrix);
std::vector<ddd>			utp_vector_matrix_multiply(const std::vector<ddd> &vector, const std::vector<std::vector<ddd>> &matrix);
std::vector<std::vector<ddd>>		transpose(const std::vector<std::vector<ddd>>& matrix);
std::vector<std::vector<ddd>>		extractBiases(const std::vector<std::vector<std::vector<ddd>>>& weights);
std::vector<std::vector<std::vector<ddd>>>	extractWeights(const std::vector<std::vector<std::vector<ddd>>>& weights);
void				InjectBiases(std::vector<std::vector<std::vector<ddd>>>& weights, const std::vector<std::vector<ddd>>& extractedBiases);
void				InjectWeights(std::vector<std::vector<std::vector<ddd>>>& weights, const std::vector<std::vector<std::vector<ddd>>>& extractedWeights);
ddd 				weightedSum(std::vector<ddd> outsideValues, std::vector<ddd> insideValues);
std::vector<ddd> 			weightedSums(std::vector<ddd> outsideValues, std::vector<std::vector<ddd>> insideValues);
std::vector<std::vector<ddd>> 		SumAll(std::vector<ddd> input, std::vector<std::vector<std::vector<ddd>>> weights);
std::vector<ddd>			NetworkRunSum(std::vector<ddd> input, std::vector<std::vector<std::vector<ddd>>> weights);
void				Copy(std::vector<ddd> *to, const std::vector<ddd> &from);
std::vector<std::vector<std::vector<ddd>>>	transpose(const std::vector<std::vector<std::vector<ddd>>>& matrix);
std::vector<ddd>			no_format_needed(std::vector<ddd> d, std::vector<ddd> last);
std::vector<ddd>			formatExpectedOutputAUDUSDCurrent(std::vector<ddd> current, std::vector<ddd> last);
ddd				UnreadableStringToDouble(std::string str);
std::string			DoubleToUnreadableString(ddd *d);
std::vector<ddd>			longUnreadableStringToArray(std::string s);
std::vector<std::string>	split(std::string str, char split);
std::vector<std::string> 	split(std::string str, std::vector<char> split);
std::string			millisToString(int64_t milliseconds);
std::string			millisToString(ddd milliseconds);
size_t				sumFor(std::vector<size_t> &v, size_t max);
size_t				max(std::vector<size_t> &v);
#endif
