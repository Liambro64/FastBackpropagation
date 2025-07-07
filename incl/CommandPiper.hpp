#pragma once
#ifndef STOCKDATAFEEDER_HPP
#define STOCKDATAFEEDER_HPP

#include "../Project.hpp"

class CommandPiper
{
private:
	FILE *stream;
	int status;
	std::string ret;
public:
	CommandPiper(std::string command, std::string mode = "r");
	~CommandPiper() {};
	std::string readOnce(size_t maxChars = PATH_MAX);
	void printOnce(size_t maxChars = PATH_MAX);
	std::string readAll(size_t maxChars = PATH_MAX);
	void printAll(size_t maxChars = PATH_MAX);
};

#endif