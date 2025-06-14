#include "../Project.hpp"
ddd LossFunction(ddd prediction, ddd expected, int size)
{
    return (prediction - expected) * (prediction - expected);
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

ddd inverseDecimal(ddd d) {
    return ((double)1) - abs(d);
}
bool compare (char c, vec<char> cs) {
    for (int i = 0; i < cs.size(); i++) {
        if (c == cs[i])
            return true;
    }
    return false;
}
size_t count(std::string str, char c)
{
    size_t i = 0;
    for (int j = 0; j < str.size(); j++)
    {
        if (str[j] == c)
            i++;
    }
    return i;
}
size_t count(std::string str, vec<char> c)
{
    size_t i = 0;
    for (int j = 0; j < str.size(); j++)
    {
        for (int k = 0; k < c.size(); k++)
        if (str[j] == c[k]) {
            i++;
            continue;
        }
    }
    return i;
}
vec<std::string> split(std::string str, char split)
{
    vec<std::string> ret(count(str, split));
    int index = 0;
    int last = 0;
    for (int i = 0; i < str.size(); i++)
    {
        if (str[i] == split)
        {
            ret[index++] = str.substr(last, i - last);
            last = i + 1;
        }
    }
    return ret;
}
vec<std::string> split(std::string str, vec<char> split)
{
    vec<std::string> ret(count(str, split));
    int index = 0;
    int last = 0;
    for (int i = 0; i < str.size(); i++)
    {
        if (compare(str[i], split))
        {
            ret[index++] = str.substr(last, i - last);
            last = i + 1;
        }
    }
    return ret;
}
vec<std::string> splitSkipN(std::string str, char c, int n)
{
    vec<std::string> ret(count(str, c));
    int index = -n;
    int last = 0;
    for (int i = 0; i < str.size(); i++)
    {
        if (str[i] == c)
        {
            if (index < 0)
            {
                index++;
                last = i + 1;
            }
            else
            {
                ret[index++] = str.substr(last, i - last);
                last = i + 1;
            }
        }
    }
    return ret;
}
vec<std::string> splitSkipN(std::string str, vec<char> split, int n)
{
    vec<std::string> ret(count(str, split) - n);
    int index = -n;
    int last = 0;
    for (int i = 0; i < str.size(); i++)
    {
        if (compare(str[i], split))
        {
            if (index < 0)
            {
                index++;
                last = i + 1;
            }
            else
            {
                ret[index++] = str.substr(last, i - last);
                last = i+1;
            }
        }
    }
    return ret;
}
vec<ddd> formatSingleAUDUSDDatapoint(std::string line)
{
    vec<std::string> splittedstr = splitSkipN(line, '	', 1);
    vec<ddd> ret(splittedstr.size());
    for (int i = 0; i < splittedstr.size(); i++)
        ret[i] = std::strtod(&(splittedstr[i][0]), nullptr);
    return ret;
}
vec<ddd> formatSingleAUDUSDDatapointCurrent(std::string line)
{
    vec<std::string> splittedstr = splitSkipN(line, {'\t', '\r'}, 2);
    vec<ddd> ret(splittedstr.size());
    for (int i = 0; i < splittedstr.size(); i++)
        ret[i] = std::strtod(&(splittedstr[i][0]), nullptr);
    ret[ret.size() - 1] = inverseDecimal(1/ret[ret.size() - 1]);
    return ret;
}
// in the csv its stored like (without spaces except for between date and time):
// date time \t open \t high \t low \t close \t volume
// to:
// high, low, close, volume
vec<vec<ddd>> formatAUDUSDData(std::ifstream *stream, int maxlines)
{
    vec<vec<ddd>> ret(maxlines == -1 ? 200000 : maxlines);
    std::string str;
    int i = 0;
    while (std::getline(*stream, str))
    {
        ret[i] = formatSingleAUDUSDDatapointCurrent(str);
        i++;
        if (maxlines != -1 && i > maxlines)
        {
            break;
        }
    }
    return ret;
}
vec<ddd> no_format_needed(vec<ddd> d, vec<ddd> last) {
    return d;
}
// high, low, close, volume -> high, low, close, volume (0->inf as 0->1), % change (as (halved)-1->1(doubled))
vec<ddd> formatExpectedOutputAUDUSDCurrent(vec<ddd> current, vec<ddd> last) {
    vec<ddd> ret(5);
    int size = current.size();
    if (ret.size() == size + 1)
        std::copy(current.begin(), current.end(), ret.begin());
    ret[size] = current[size - 2] / last[size - 2] - 1;
    return ret;
}
ddd LossDerivative(ddd prediction, ddd expected)
{
    return 2 * (expected - prediction);
}
ddd sigmoid(ddd x)
{
    return 1 / (1 + exp(-x));
}
ddd sigmoidDerivative(ddd x)
{
    return x * (1 - x);
}
size_t max(vec<size_t> &v)
{
    size_t maxVal = 0;
    for (size_t i = 0; i < v.size(); i++)
    {
        if (v[i] > maxVal)
        {
            maxVal = v[i];
        }
    }
    return maxVal;
}
size_t min(size_t a, size_t b)
{
    return (a < b) ? a : b;
}
size_t sumFor(vec<size_t> &v, size_t max)
{
    size_t sum = 0;
    for (size_t i = 0; i < min(v.size(), max); i++)
    {
        sum += v[i];
    }
    return sum;
}
// Vector addition
vec<ddd> add(const vec<ddd> &v1, const vec<ddd> &v2)
{
    if (v1.size() != v2.size())
    {
        std::cerr << "Error: Vector sizes do not match for addition." << std::endl;
        return {};
    }
    vec<ddd> result(v1.size());
    for (size_t i = 0; i < v1.size(); ++i)
    {
        result[i] = v1[i] + v2[i];
    }
    return result;
}

// Vector subtraction (v1 - v2)
vec<ddd> subtract(const vec<ddd> &v1, const vec<ddd> &v2)
{
    if (v1.size() != v2.size())
    {
        throw "Error: Matrix sizes do not match for subtraction";
        return {};
    }
    vec<ddd> result(v1.size());
    for (size_t i = 0; i < v1.size(); ++i)
    {
        result[i] = v1[i] - v2[i];
    }
    return result;
}
ddd dot_product(const vec<ddd> &v1, const vec<ddd> &v2)
{
    if (v1.size() != v2.size())
    {
        // In a real implementation, handle errors properly
        std::cerr << "Error: Vector sizes do not match for dot product." << std::endl;
        return 0.0;
    }
    ddd result = 0.0;
    for (size_t i = 0; i < v1.size(); ++i)
    {
        result += v1[i] * v2[i];
    }
    return result;
}
// Matrix addition
vec<vec<ddd>> add(const vec<vec<ddd>> &m1, const vec<vec<ddd>> &m2)
{
    if (m1.size() != m2.size() || (m1.empty() ? false : m1[0].size() != m2[0].size()))
    {
        std::cerr << "Error: Matrix sizes do not match for addition." << std::endl;
        return {};
    }
    if (m1.empty())
        return {};
    vec<vec<ddd>> result(m1.size(), vec<ddd>(m1[0].size()));
    for (size_t i = 0; i < m1.size(); ++i)
    {
        for (size_t j = 0; j < m1[0].size(); ++j)
        {
            result[i][j] = m1[i][j] + m2[i][j];
        }
    }
    return result;
}

// Matrix subtraction (m1 - m2)
vec<vec<ddd>> subtract(const vec<vec<ddd>> &m1, const vec<vec<ddd>> &m2)
{
    if (m1.size() != m2.size() || (m1.empty() ? false : m1[0].size() != m2[0].size()))
    {
        throw "Error: Matrix sizes do not match for subtraction";
        return {};
    }
    if (m1.empty())
        return {};
    vec<vec<ddd>> result(m1.size(), vec<ddd>(m1[0].size()));
    for (size_t i = 0; i < m1.size(); ++i)
    {
        for (size_t j = 0; j < m1[0].size(); ++j)
        {
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
vec<ddd> scalar_multiply(ddd scalar, const vec<ddd> &vector)
{
    vec<ddd> result(vector.size());
    for (size_t i = 0; i < vector.size(); ++i)
    {
        result[i] = scalar * vector[i];
    }
    return result;
}
vec<vec<ddd>> scalar_multiply(ddd scalar, const vec<vec<ddd>> &matrix)
{
    if (matrix.empty())
        return {};
    vec<vec<ddd>> result(matrix.size(), vec<ddd>(matrix[0].size()));
    for (size_t i = 0; i < matrix.size(); ++i)
    {
        for (size_t j = 0; j < matrix[0].size(); ++j)
        {
            result[i][j] = scalar * matrix[i][j];
        }
    }
    return result;
}
vec<ddd> matrix_vector_multiply(const vec<vec<ddd>> &matrix, const vec<ddd> &vector)
{
    if (matrix.empty() || (matrix[0].size() != vector.size()))
    {
        std::cerr << "Error: Matrix/vector sizes do not match for matrix-vector multiply." << std::endl;
        return {};
    }
    vec<ddd> result(matrix.size(), 0.0);
    for (size_t i = 0; i < matrix.size(); ++i)
    {
        result[i] = dot_product(matrix[i], vector);
    }
    return result;
}

// Vector-matrix multiplication (vector * matrix) - assuming row vector * matrix
vec<ddd> vector_matrix_multiply(const vec<ddd> &vector, const vec<vec<ddd>> &matrix)
{
    if (matrix.empty() || matrix[0].size() != vector.size())
    {
        std::cerr << "Error: Vector/matrix sizes do not match for vector-matrix multiply." << std::endl;
        return {};
    }
    vec<ddd> result(matrix[0].size(), 0.0);
    for (size_t j = 0; j < matrix[0].size(); ++j)
    {
        for (size_t i = 0; i < matrix.size(); ++i)
        {
            result[j] += vector[j] * matrix[i][j];
        }
    }
    return result;
}
vec<vec<ddd>> transpose(const vec<vec<ddd>> &matrix)
{
    if (matrix.empty())
        return {};
    vec<vec<ddd>> result(matrix[0].size(), vec<ddd>(matrix.size()));
    for (size_t i = 0; i < matrix.size(); ++i)
    {
        for (size_t j = 0; j < matrix[0].size(); ++j)
        {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}
vec<vec<vec<ddd>>> transpose(const vec<vec<vec<ddd>>> &matrix)
{
    if (matrix.empty() || matrix[0].empty())
        return {};
    vec<vec<vec<ddd>>> result(matrix.size());
    for (size_t k = 0; k < matrix.size(); ++k)
    {
        result[k].resize(matrix[k][0].size(), vec<ddd>(matrix[k].size()));
        for (size_t i = 0; i < matrix[k].size(); ++i)
        {
            for (size_t j = 0; j < matrix[k][0].size(); ++j)
            {
                result[k][j][i] = matrix[k][i][j];
            }
        }
    }
    return result;
}
vec<vec<vec<ddd>>> extractWeights(const vec<vec<vec<ddd>>> &weights)
{
    vec<vec<vec<ddd>>> extractedWeights;
    for (int i = 0; i < weights.size(); i++)
    {
        vec<vec<ddd>> extractedLayer;
        for (int j = 0; j < weights[i].size(); j++)
        {
            vec<ddd> extractedNeuron(weights[i][j].size() - 1);
            std::copy(weights[i][j].begin() + 1, weights[i][j].end(), extractedNeuron.begin());
            extractedLayer.push_back(extractedNeuron);
        }
        extractedWeights.push_back(extractedLayer);
    }
    return extractedWeights;
}
void InjectWeights(vec<vec<vec<ddd>>> &weights, const vec<vec<vec<ddd>>> &extractedWeights)
{
    for (int i = 0; i < weights.size(); i++)
    {
        for (int j = 0; j < weights[i].size(); j++)
        {
            for (int k = 1; k < weights[i][j].size(); k++)
            {
                weights[i][j][k] = extractedWeights[i][j][k - 1]; // Adjust index for bias
            }
        }
    }
}

vec<vec<ddd>> extractBiases(const vec<vec<vec<ddd>>> &weights)
{
    vec<vec<ddd>> extractedWeights;
    for (int i = 0; i < weights.size(); i++)
    {
        vec<ddd> extractedLayer(weights[i].size());
        for (int j = 0; j < weights[i].size(); j++)
        {
            extractedLayer[j] = weights[i][j][0]; // Bias is always at index 0
        }
        extractedWeights.push_back(extractedLayer);
    }
    return extractedWeights;
}
void InjectBiases(vec<vec<vec<ddd>>> &weights, const vec<vec<ddd>> &extractedBiases)
{
    for (int i = 0; i < weights.size(); i++)
    {
        for (int j = 0; j < weights[i].size(); j++)
        {
            weights[i][j][0] = extractedBiases[i][j]; // Bias is always at index 0
        }
    }
}
// specifically for running the network
ddd weightedSum(vec<ddd> outsideValues, vec<ddd> insideValues)
{
    if (outsideValues.size() != insideValues.size() - 1)
    {
        std::cerr << "Error: Vector sizes do not match for weighted sum. Sizes are: " << outsideValues.size() << ", " << insideValues.size() << std::endl;
        return {};
    }
    ddd sum = insideValues[0]; // Start with the bias
    for (size_t i = 0; i < outsideValues.size(); i++)
    {
        sum += outsideValues[i] * insideValues[i + 1];
    }
    return sigmoid(sum);
}
vec<ddd> weightedSums(vec<ddd> outsideValues, vec<vec<ddd>> insideValues)
{
    vec<ddd> sums(insideValues.size());
    for (size_t i = 0; i < insideValues.size(); i++)
    {
        sums[i] = weightedSum(outsideValues, insideValues[i]);
    }
    return sums;
}
vec<vec<ddd>> keepSum(vec<ddd> input, vec<vec<vec<ddd>>> weights)
{
    vec<vec<ddd>> sums(weights.size());
    for (size_t i = 0; i < weights.size(); i++)
    {
        sums[i] = weightedSums((i == 0 ? input : sums[i - 1]), weights[i]);
    }
    return sums;
}
// does everything and only keeps the end
vec<ddd> NetworkRunSum(vec<ddd> input, vec<vec<vec<ddd>>> weights)
{
    vec<ddd> value(input.size());
    std::copy(input.begin(), input.end(), value.begin());
    for (int i = 0; i < weights.size(); i++)
    {
        value = weightedSums(value, weights[i]);
    }
    return value;
}
void Copy(vec<ddd> *to, const vec<ddd> &from)
{
    if (to->size() != from.size())
    {
        to->resize(from.size());
    }
    std::copy(from.begin(), from.end(), to->begin());
}
std::string millisToString(int64_t milliseconds)
{
    int64_t seconds = (milliseconds / 1000);
    milliseconds %= 1000;
    int64_t minutes = seconds / 60;
    seconds %= 60;
    int64_t hours = minutes / 60;
    minutes %= 60;

    std::string result;
    if (hours > 0)
    {
        result += std::to_string(hours) + "h:";
    }
    if (minutes > 0 || hours > 0)
    {
        result += std::to_string(minutes) + "m:";
    }
    result += std::to_string(seconds) + "s:";
    result += std::to_string(milliseconds) + "ms";

    return result;
}
std::string millisToString(double milliseconds)
{
    int64_t seconds = (milliseconds / 1000);
    int millis = (int)fmod(milliseconds, 1000);
    int64_t minutes = seconds / 60;
    seconds %= 60;
    int64_t hours = minutes / 60;
    minutes %= 60;

    std::string result;
    if (hours > 0)
    {
        result += std::to_string(hours) + "h:";
    }
    if (minutes > 0 || hours > 0)
    {
        result += std::to_string(minutes) + "m:";
    }
    result += std::to_string(seconds) + "s:";
    result += std::to_string(millis) + "ms";

    return result;
}