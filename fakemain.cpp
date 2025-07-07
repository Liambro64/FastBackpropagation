#include "Project.hpp"

void func(FILE **streamptr)
{
	std::string str = "";
	str.resize(2048);
	*streamptr = popen("bash -c './cmd.txt'", "r");
	
	sleep(5);
	fgets(str.data(), 2047, *streamptr);
	std::fstream outFile = std::fstream("./cmdout.txt");
	outFile.write(str.data(), str.size());
	std::cout << str << std::endl;
}

int main()
{
	bool waiting = true;
	FILE **ptr = (FILE **)malloc(sizeof(FILE **));
	auto result = std::async(std::launch::async, [ptr]()
							 { func(ptr); });
	std::chrono::duration<double, std::milli> wait(3000);
	while (waiting)
	{
		auto status = result.wait_for(wait);
		switch (status)
		{
		case std::future_status::timeout:
			std::cout << "Timeout" << std::endl;
			break;
		case std::future_status::ready:
			std::cout << "Done" << std::endl;
			break;
		case std::future_status::deferred:
			std::cout << "Not started" << std::endl;
			break;
		}
	}
	free(ptr);
}