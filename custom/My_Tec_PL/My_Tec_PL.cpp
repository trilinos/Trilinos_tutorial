#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardParameterEntryValidators.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_Version.hpp"

int main(int argc, char* argv[])
{

  std::cout << Teuchos::Teuchos_Version() << std::endl << std::endl;

  // Creating an empty parameter list looks like:
  Teuchos::ParameterList Command_List;

  // setting up the command list 
  Command_List.set("Package Suite", "examples", "the string name of the package to be run");
  Command_List.set("Example", "Epetra_Simple_Vector", "the string name of the specific example to run");
  Command_List.set("Num procs", (int)4, "the number of processors to run the given examples with");

  
std::string package = " "+Command_List.get<std::string>("Package Suite");
std::string example = " "+Command_List.get<std::string>("Example");
std::string numprocs = " "+Command_List.get<int>("Num procs");
std::string command = "so the command becomes: "+"Package= "+package+" Example= "+example+" Number of threads is: "+numprocs;
char* line = (char*)command.c_str();
//system(line);
std::cout << command;

  std::cout << "\n# Printing this parameter list using operator<<(...) ...\n\n";
  std::cout << Command_List << std::endl;

  std::cout << "\n# Printing the parameter list only showing documentation fields ...\n\n";
    Command_List.print(std::cout,Teuchos::ParameterList::PrintOptions().showDoc(true).indent(2).showTypes(true));

  return 0;
}
