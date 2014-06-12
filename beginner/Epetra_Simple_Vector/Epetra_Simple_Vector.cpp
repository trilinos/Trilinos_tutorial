#include "Epetra_SerialComm.h"
#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_Version.h"
#include "../../aprepro_vhelp.h"

int main(int argc, char *argv[])
{

  std::cout << Epetra_Version() << std::endl << std::endl;

  Epetra_SerialComm Comm;

  int NumElements = 1000;

  // Construct a Map with NumElements and index base of 0
  Epetra_Map Map(NumElements, 0, Comm);

  // Create x and b vectors
  Epetra_Vector x(Map);
  Epetra_Vector b(Map);

  b.Random();

  x.Update(2.0, b, 0.0); // x = 2*b

  double bnorm, xnorm;

  x.Norm2(&xnorm);
  b.Norm2(&bnorm);

  std::cout << "2 norm of x = " << xnorm << std::endl
       << "2 norm of b = " << bnorm << std::endl;

  return 0;
}


