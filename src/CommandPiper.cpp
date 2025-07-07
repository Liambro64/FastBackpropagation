#include "../Project.hpp"

CommandPiper::CommandPiper(std::string command, std::string mode)
{
	ret.resize(PATH_MAX);
	stream = popen(command.data(), mode.data());
	if (stream == nullptr)
		throw std::invalid_argument("Couldn't run command");
	status = 0;
}

std::string CommandPiper::readOnce(size_t maxChars)
{
	ret.resize(maxChars + 1);
	return fgets(ret.data(), maxChars, stream);
}
std::string CommandPiper::readAll(size_t maxChars)
{
	std::string ret = "";
	while (fgets(this->ret.data(), maxChars, stream))
		ret.append(this->ret.data());
	return ret;
}
void CommandPiper::printOnce(size_t maxChars)
{
	ret.resize(maxChars + 1);
	std::cout << fgets(ret.data(), maxChars, stream) << std::endl;
}
void CommandPiper::printAll(size_t maxChars)
{
	ret.resize(maxChars + 1);
	while (fgets(ret.data(), maxChars, stream))
	{
		std::cout << ret;
		ret.clear();
		ret.resize(maxChars + 1);
	}
	std::cout << std::endl;
}