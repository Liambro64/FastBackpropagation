#include "../Project.hpp"
ddd LossFunction(ddd prediction, ddd expected, int size)
{
	return (prediction - expected) * (prediction - expected) / size;
}
ddd LossFunction(vec<ddd> prediction, vec<ddd> expected)
{
	auto error = subtract(prediction, expected);
	ddd sum = 0;
	for (int i = 0; i < error.size(); i++)
	{
		sum += error[i] * error[i];
	}
	return (sum) / error.size();

}
ddd LossDerivative(ddd prediction, ddd expected)
{
	return 2 * (prediction - expected);
}
ddd sigmoid(ddd x)
{
	return 1 / (1 + exp(-x));
}
ddd sigmoidDerivative(ddd x)
{
	return x * (1 - x);
}
size_t max(vec<size_t> &v) {
    size_t maxVal = 0;
    for (size_t i = 0; i < v.size(); i++) {
        if (v[i] > maxVal) {
            maxVal = v[i];
        }
    }
    return maxVal;
}
size_t min(size_t a, size_t b) {
    return (a < b) ? a : b;
}
size_t sumFor(vec<size_t> &v, size_t max) {
    size_t sum = 0;
    for (size_t i = 0; i < min(v.size(), max); i++) {
        sum += v[i];
    }
    return sum;
}
// Vector addition
vec<ddd> add(const vec<ddd>& v1, const vec<ddd>& v2) {
    if (v1.size() != v2.size()) {
         std::cerr << "Error: Vector sizes do not match for addition." << std::endl;
        return {};
    }
    vec<ddd> result(v1.size());
    for (size_t i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] + v2[i];
    }
    return result;
}

// Vector subtraction (v1 - v2)
vec<ddd> subtract(const vec<ddd>& v1, const vec<ddd>& v2) {
    if (v1.size() != v2.size()) {
         std::cerr << "Error: Vector sizes do not match for subtraction." << std::endl;
        return {};
    }
    vec<ddd> result(v1.size());
    for (size_t i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] - v2[i];
    }
    return result;
}
ddd dot_product(const vec<ddd>& v1, const vec<ddd>& v2) {
    if (v1.size() != v2.size()) {
        // In a real implementation, handle errors properly
        std::cerr << "Error: Vector sizes do not match for dot product." << std::endl;
        return 0.0;
    }
    ddd result = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}
// Matrix addition
vec<vec<ddd>> add(const vec<vec<ddd>>& m1, const vec<vec<ddd>>& m2) {
    if (m1.size() != m2.size() || (m1.empty() ? false : m1[0].size() != m2[0].size())) {
         std::cerr << "Error: Matrix sizes do not match for addition." << std::endl;
        return {};
    }
    if (m1.empty()) return {};
    vec<vec<ddd>> result(m1.size(), vec<ddd>(m1[0].size()));
    for (size_t i = 0; i < m1.size(); ++i) {
        for (size_t j = 0; j < m1[0].size(); ++j) {
            result[i][j] = m1[i][j] + m2[i][j];
        }
    }
    return result;
}

// Matrix subtraction (m1 - m2)
vec<vec<ddd>> subtract(const vec<vec<ddd>>& m1, const vec<vec<ddd>>& m2) {
    if (m1.size() != m2.size() || (m1.empty() ? false : m1[0].size() != m2[0].size())) {
         std::cerr << "Error: Matrix sizes do not match for subtraction." << std::endl;
        return {};
    }
     if (m1.empty()) return {};
    vec<vec<ddd>> result(m1.size(), vec<ddd>(m1[0].size()));
    for (size_t i = 0; i < m1.size(); ++i) {
        for (size_t j = 0; j < m1[0].size(); ++j) {
            result[i][j] = m1[i][j] - m2[i][j];
        }
    }
    return result;
}
vec<vec<ddd>> outerProduct(const vec<ddd> &a, const vec<ddd> &b)
{
	vec<vec<ddd>> result(a.size(), vec<ddd>(b.size()));
	for (size_t i = 0; i < a.size(); i++)
	{
		for (size_t j = 0; j < b.size(); j++)
		{
			result[i][j] = a[i] * b[j];
		}
	}
	return result;
}
vec<ddd> scalar_multiply(ddd scalar, const vec<ddd>& vector) {
    vec<ddd> result(vector.size());
    for (size_t i = 0; i < vector.size(); ++i) {
        result[i] = scalar * vector[i];
    }
    return result;
}
vec<vec<ddd>> scalar_multiply(ddd scalar, const vec<vec<ddd>>& matrix) {
    if (matrix.empty()) return {};
    vec<vec<ddd>> result(matrix.size(), vec<ddd>(matrix[0].size()));
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[0].size(); ++j) {
            result[i][j] = scalar * matrix[i][j];
        }
    }
    return result;
}
vec<ddd> matrix_vector_multiply(const vec<vec<ddd>>& matrix, const vec<ddd>& vector) {
    if (matrix.empty() || (matrix[0].size() != vector.size())) {
        std::cerr << "Error: Matrix/vector sizes do not match for matrix-vector multiply." << std::endl;
        return {};
    }
    vec<ddd> result(matrix.size(), 0.0);
    for (size_t i = 0; i < matrix.size(); ++i) {
        result[i] = dot_product(matrix[i], vector);
    }
    return result;
}

// Vector-matrix multiplication (vector * matrix) - assuming row vector * matrix
vec<ddd> vector_matrix_multiply(const vec<ddd>& vector, const vec<vec<ddd>>& matrix) {
     if (matrix.empty() ||  matrix[0].size() != vector.size()) {
        std::cerr << "Error: Vector/matrix sizes do not match for vector-matrix multiply." << std::endl;
        return {};
    }
    vec<ddd> result(matrix[0].size(), 0.0);
    for (size_t j = 0; j < matrix[0].size(); ++j) {
        for (size_t i = 0; i < matrix.size(); ++i) {
            result[j] += vector[j] * matrix[i][j];
        }
    }
    return result;
}
vec<vec<ddd>> transpose(const vec<vec<ddd>>& matrix) {
    if (matrix.empty()) return {};
    vec<vec<ddd>> result(matrix[0].size(), vec<ddd>(matrix.size()));
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[0].size(); ++j) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}
vec<vec<vec<ddd>>> transpose(const vec<vec<vec<ddd>>>& matrix) {
    if (matrix.empty() || matrix[0].empty()) return {};
    vec<vec<vec<ddd>>> result(matrix.size());
    for (size_t k = 0; k < matrix.size(); ++k) {
        result[k].resize(matrix[k][0].size(), vec<ddd>(matrix[k].size()));
        for (size_t i = 0; i < matrix[k].size(); ++i) {
            for (size_t j = 0; j < matrix[k][0].size(); ++j) {
                result[k][j][i] = matrix[k][i][j];
            }
        }
    }
    return result;
}
vec<vec<vec<ddd>>> extractWeights(const vec<vec<vec<ddd>>>& weights) {
	vec<vec<vec<ddd>>> extractedWeights;
	for (int i = 0; i < weights.size(); i++) {
		vec<vec<ddd>> extractedLayer;
		for (int j = 0; j < weights[i].size(); j++) {
			vec<ddd> extractedNeuron(weights[i][j].size() - 1);
            std::copy(weights[i][j].begin() + 1, weights[i][j].end(), extractedNeuron.begin()); 
            extractedLayer.push_back(extractedNeuron);
		}
		extractedWeights.push_back(extractedLayer);
	}
	return extractedWeights;
}
void InjectWeights(vec<vec<vec<ddd>>>& weights, const vec<vec<vec<ddd>>>& extractedWeights) {
	for (int i = 0; i < weights.size(); i++) {
		for (int j = 0; j < weights[i].size(); j++) {
			for (int k = 1; k < weights[i][j].size(); k++) {
				weights[i][j][k] = extractedWeights[i][j][k - 1]; // Adjust index for bias
			}
		}
	}
}

vec<vec<ddd>> extractBiases(const vec<vec<vec<ddd>>>& weights) {
	vec<vec<ddd>> extractedWeights;
	for (int i = 0; i < weights.size(); i++) {
		vec<ddd> extractedLayer(weights[i].size());
		for (int j = 0; j < weights[i].size(); j++) {
			extractedLayer[j] = weights[i][j][0]; // Bias is always at index 0
		}
		extractedWeights.push_back(extractedLayer);
	}
	return extractedWeights;
}
void InjectBiases(vec<vec<vec<ddd>>>& weights, const vec<vec<ddd>>& extractedBiases) {
	for (int i = 0; i < weights.size(); i++) {
		for (int j = 0; j < weights[i].size(); j++) {
			weights[i][j][0] = extractedBiases[i][j]; // Bias is always at index 0
		}
	}
}
//specifically for running the network
ddd weightedSum(vec<ddd> outsideValues, vec<ddd> insideValues) {
	if (outsideValues.size() != insideValues.size() - 1) {
		std::cerr << "Error: Vector sizes do not match for weighted sum. Sizes are: " << outsideValues.size() << ", " << insideValues.size() << std::endl;
		return {};
	}
	ddd sum = insideValues[0]; // Start with the bias
	for (size_t i = 0; i < outsideValues.size(); i++) {
		sum += outsideValues[i] * insideValues[i + 1];
	}
	return sigmoid(sum);
}
vec<ddd> weightedSums(vec<ddd> outsideValues, vec<vec<ddd>> insideValues) {
	vec<ddd> sums(insideValues.size());
	for (size_t i = 0; i < insideValues.size(); i++) {
		sums[i] = weightedSum(outsideValues, insideValues[i]);
	}
	return sums;
}
vec<vec<ddd>> keepSum(vec<ddd> input, vec<vec<vec<ddd>>> weights) {
	vec<vec<ddd>> sums(weights.size());
	for (size_t i = 0; i < weights.size(); i++) {
		sums[i] = weightedSums((i == 0 ? input : sums[i - 1]), weights[i]);
	}
	return sums;
}
//does everything and only keeps the end
vec<ddd> NetworkRunSum(vec<ddd> input, vec<vec<vec<ddd>>> weights) {
	vec<ddd> value(input.size());
	std::copy(input.begin(), input.end(), value.begin());
	for (int i = 0; i < weights.size(); i++) {
		value = weightedSums(value, weights[i]);
	}
	return value;
}
void Copy(vec<ddd> *to, const vec<ddd> &from) {
    if (to->size() != from.size()) {
        to->resize(from.size());
    }
    std::copy(from.begin(), from.end(), to->begin());
}
std::string milisToString(int64_t milliseconds) {
    int64_t seconds =( milliseconds / 1000);
    milliseconds %= 1000;
    int64_t minutes = seconds / 60;
    seconds %= 60;
    int64_t hours = minutes / 60;
    minutes %= 60;

    std::string result;
    if (hours > 0) {
        result += std::to_string(hours) + "h:";
    }
    if (minutes > 0 || hours > 0) {
        result += std::to_string(minutes) + "m:";
    }
    result += std::to_string(seconds) + "s:";
    result += std::to_string(milliseconds) + "ms";
    
    return result;
}
std::string milisToString(double milliseconds) {
    int64_t seconds =(milliseconds / 1000);
    int millis = (int)fmod(milliseconds, 1000);
    int64_t minutes = seconds / 60;
    seconds %= 60;
    int64_t hours = minutes / 60;
    minutes %= 60;

    std::string result;
    if (hours > 0) {
        result += std::to_string(hours) + "h:";
    }
    if (minutes > 0 || hours > 0) {
        result += std::to_string(minutes) + "m:";
    }
    result += std::to_string(seconds) + "s:";
    result += std::to_string(millis) + "ms";
    
    return result;
}