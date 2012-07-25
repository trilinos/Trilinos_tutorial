#include "Optika_GUI.hpp"
#include "../../aprepro_vhelp.h"

std::string convertInt(int number);

int main(int argc, char *argv[])
{
	using std::string;
	using Optika::OptikaGUI;
	using Teuchos::RCP;
	using Teuchos::ParameterList;
	using Teuchos::rcp;

	RCP<ParameterList> userInput = rcp(new ParameterList);
	Optika::getInput("gui.xml", userInput);

  int numNodesX = 5;
	numNodesX = Teuchos::getParameter<int>(*userInput, "numNodesX");
  int numNodesY = 5;
	numNodesY = Teuchos::getParameter<int>(*userInput, "numNodesY");
  int numProcsX = 1;
	numProcsX = Teuchos::getParameter<int>(*userInput, "numProcsX");
  int numProcsY = 1;
	numProcsY = Teuchos::getParameter<int>(*userInput, "numProcsY");
  int numPoints = 25;
	numPoints = Teuchos::getParameter<int>(*userInput, "numPoints");
//cout << "NNX: " << numNodesX << " NNY: " << numNodesY << " NPX: " << numProcsX << " NPY: " << numProcsY << " NPTS: " << numPoints << endl;
	const string typeParam = "Report Type";
	std::string type = Teuchos::getParameter<string>(*userInput, typeParam);

if (type == "verbose") type = " -v";
else if (type =="summary") type = " -s";
else type = "";

string command = "mpiexec -np "+convertInt(numProcsX*numProcsY)+" Epetra_Basic_Perf "+convertInt(numNodesX)+" "+convertInt(numNodesY)+" "+convertInt(numProcsX)+" "+convertInt(numProcsY)+" "+convertInt(numPoints)+type;
char* line = (char*)command.c_str();
/**
char* args[argn];
args[0] = (char*)command.c_str();
args[1] = (char*)convertInt(numNodesX).c_str();
args[2] = (char*)convertInt(numNodesY).c_str();
args[3] = (char*)convertInt(numProcsX).c_str();
args[4] = (char*)convertInt(numProcsY).c_str();
args[5] = (char*)convertInt(numPoints).c_str();
args[6] = (char*)type.c_str();
/**/
//char* args[7] = {(char*)command.c_str(), (char*)convertInt(numNodesX).c_str(), (char*)convertInt(numNodesY).c_str(), (char*)convertInt(numProcsX).c_str(), (char*)convertInt(numProcsY).c_str(), (char*)convertInt(numPoints).c_str(), (char*)type.c_str()};

//cout << "TOP"<<endl<<"arg0: " << args[0]<< "arg1: " << args[1]<< ", arg2: " << args[2]<< ", arg3: " << args[3]<< ", arg4: " << args[4]<< ", arg5: " << args[5]<< ", arg6: " << args[6] << endl;

system(line);
return 0;

}
std::string convertInt(int number)
{
	std::stringstream ss;
	ss << number;
	return ss.str();
}

