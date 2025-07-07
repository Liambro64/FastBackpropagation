#include "Project.hpp"

int main(int argc, char** argv)
{
	CommandPiper p = CommandPiper("./script.sh");
	p.printAll();
	return 0;
}