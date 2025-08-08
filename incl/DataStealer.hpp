#pragma once
#ifndef STOCKDATAFEEDER_HPP
#define STOCKDATAFEEDER_HPP

#include "../Project.hpp"

//do not use

class DataStealer
{
private:

public:
	size_t (*callback)(char *, size_t, size_t, void *);
	DataStealer(size_t (*)(char *, size_t, size_t, void *));
	std::pair<std::string, std::string> LoadFromFile(std::string infile, std::string rtfile = "cmd.cpp.txt");
	void ConnectToStream(CURLcode *ret, std::string file);
	static std::function<size_t(char *, size_t, size_t, void *)> callbackWrapper(DataStealer *ds);
};

#endif