#include "Optika_GUI.hpp"

#include "../../aprepro_vhelp.h"

std::string gui_base;

/**
  *This method converts an integer value into a string by using a string stream
  **/
std::string convertInt(int number)
{
	std::stringstream ss;
	ss << number;
	return ss.str();
}
/**
  *This method converts an double value into a string by using a string stream
  **/
std::string convertDouble(double number)
{
	std::stringstream ss;
	ss << number;
	return ss.str();
}

/**
  *This is the Main method for the Gui. this is where all of the action happens
  * In this example the interactions are a little strange since the inputs are actually for command line arguments however any type of action can be executed here.
  **/
void runExample(Teuchos::RCP<const Teuchos::ParameterList> userInput){
	using std::string;
	using Optika::OptikaGUI;
	using Teuchos::RCP;
	using Teuchos::ParameterList;
	using Teuchos::rcp;

	// these lines get the input from the gui using the method from Teuchos. it requires the ParameterList and the name of the parameter
	const string a = "Example Suite";
	std::string package = " "+Teuchos::getParameter<string>(*userInput, a);
	const string c = "Number of threads";
	std::string numprocs = " "+convertInt(Teuchos::getParameter<int>(*userInput, c));
	std::string command = "";
	if(gui_base == "cmake") command = "./cmake_guirun.sh";
	else 			command = "./guirun.sh";

	command = command + package;
	if(package == " beginner"){
		const string b = "Beginner Example";
		std::string example = " "+Teuchos::getParameter<string>(*userInput, b);
	command = command + example + numprocs;
	}
	else if(package == " advanced"){
		const string b = "Advanced Example";
		std::string example = " "+Teuchos::getParameter<string>(*userInput, b);
		std::string exeargs = " ";
		Teuchos::ParameterList sub = Teuchos::getParameter<Teuchos::ParameterList>(*userInput, "Advanced Example Parameters");
		if(example == " Epetra_Basic_Perf"){
			// get parameters from the sub-parameterList
			int xnodes = Teuchos::getParameter<int>(sub, "numNodesX");
			int ynodes = Teuchos::getParameter<int>(sub, "numNodesY");
			int xprocs = Teuchos::getParameter<int>(sub, "numProcsX");
			int yprocs = Teuchos::getParameter<int>(sub, "numProcsY");
			std::string numpts = Teuchos::getParameter<std::string>(sub, "numPoints");
			// set correct numprocs for mpirun
			numprocs = " "+convertInt(xprocs*yprocs);
			// get correct report type
			std::string report = Teuchos::getParameter<string>(sub, "Report Type");
			if( report == "verbose") report = "-v";
			else if(report == "summary") report = "-s";
			else report = "";
			// create argument string
			exeargs = exeargs 
				+"\""+convertInt(xnodes)+" "+convertInt(ynodes)
				+" "+convertInt(xprocs)+" "+convertInt(yprocs)
				+" "+numpts+" "+report+"\"";
		}
		else if (example == " Stratimikos_Solver_Driver"){
			writeParameterListToXmlFile(Teuchos::getParameter<ParameterList>(sub, "SD Parameters"), "gui_params.xml");
			if(gui_base == "cmake")	system("mv gui_params.xml cmake_build/advanced/Stratimikos_Solver_Driver/gui_params.xml");	
			else 			system("mv gui_params.xml advanced/Stratimikos_Solver_Driver/gui_params.xml");	
			exeargs = exeargs+"\"--input-file=gui_params.xml\"";
		}
		else if (example == " CurlLSFEM_example"){
			exeargs = exeargs + Teuchos::getParameter<std::string>(sub, "Curl Input File");
		}else if (example == " DivLSFEM_example"){
			exeargs = exeargs + Teuchos::getParameter<std::string>(sub, "Div Input File");
		}else if (example == " Stratimikos_Preconditioner"){
			system("echo Preconditioner here");
/**/
			
			string pfile =" --params-file="
			+Teuchos::getParameter<std::string>(sub, "Preconditioner Input File");
			string epfile=Teuchos::getParameter<std::string>(sub, "Preconditioner extra File");
			if(epfile == "none") epfile = "";		
			else epfile=" --extra-params-file="+epfile;
			const string prec=" --use-"+Teuchos::getParameter<std::string>(sub, "Preconditioner")+"-prec";
			string ip1="";
			if(Teuchos::getParameter<std::string>(sub, "Invert P1")=="yes"){
				ip1=" --invert-P1";
			}else ip1="";
			const string tol=" --solve-tol="
			+convertDouble(Teuchos::getParameter<double>(sub, "Tolerance"));
			string show ="";			
			if(Teuchos::getParameter<std::string>(sub, "Show Parameters")=="yes"){
				show=" --show-params";
			}else show=" --no-show-params";
			exeargs = exeargs +
				"\""+pfile+epfile+prec+ip1+tol+show+"\"";
/**/
		}
		else { 
			std::cout << "oops, invalid example somehow";
		}
		command = command + example + numprocs + exeargs;
	}
	else command = command + numprocs;

	char* line = (char*)command.c_str();
	system("pwd");
	std::string ghost = "echo "+command;
	char* bat = (char*)ghost.c_str();
	system(bat);
	system(line);
}

/**
  * This is the main method of the gui where the entire gui is set up
  *	will run the cmake build if executed with 'cmake' as an argument 
  **/
int main(int argc, char* argv[]){
	using std::string;
	using Optika::OptikaGUI;
	using Teuchos::RCP;
	using Teuchos::ParameterList;
	using Teuchos::rcp;

	//this saves the first argument passed to the gui. Since if the argument 'cmake' is passed it will use the executables in the cmake build
	gui_base="";
	if(argc >1) gui_base = gui_base+argv[1]; //preventing segfault
	
	// initializing the ParameterList
	RCP<ParameterList> userInput = rcp(new ParameterList);
	// This method creates the gui and waits for user input. it takes a path to the .xml document that formats the gui and the RCP that points to the ParameterList. It also can take an address of a function (as it does here) that will be executed for every press of the submit button
	Optika::getInput("gui/tutorial_gui/gui_input.xml", userInput, &runExample);

	// if there was no address to a function the code would continue executing from here after recieving the inputs from the gui. this execution begins after the gui closes once the submit button is pressed.

	// return at the end of execution
	return 0;
}
