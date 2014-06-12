//
// Use ML to build a smoothed aggregation multigrid operator.
// Use the operator as a black-box preconditioner in AztecOO's CG.
//
#include "Epetra_ConfigDefs.h"
#ifdef HAVE_MPI
#  include "mpi.h"
#  include "Epetra_MpiComm.h"
#else
#  include "Epetra_SerialComm.h"
#endif
#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_LinearProblem.h"
#include "Epetra_Time.h"
#include "AztecOO.h"

#include "../../aprepro_vhelp.h"

// The ML include file required when working with Epetra objects.
#include "ml_epetra_preconditioner.h"

#include "Trilinos_Util_CrsMatrixGallery.h"

using namespace Teuchos;
using namespace Trilinos_Util;

#include <iostream>

int 
main (int argc, char *argv[])
{
#ifdef EPETRA_MPI
  MPI_Init (&argc,&argv);
  Epetra_MpiComm Comm (MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif

  Epetra_Time Time(Comm);

  // Initialize a Gallery object, for generating a 3-D Laplacian
  // matrix distributed over the given communicator Comm.
  CrsMatrixGallery Gallery("laplace_3d", Comm);

  Gallery.Set("problem_size", 1000);

  // Get pointers to the generated matrix and a test linear problem.
  Epetra_RowMatrix* A = Gallery.GetMatrix();

  Epetra_LinearProblem* Problem = Gallery.GetLinearProblem();

  // Construct an AztecOO solver object for this problem.
  AztecOO solver (*Problem);

  // Create the preconditioner object and compute the multilevel hierarchy.
  ML_Epetra::MultiLevelPreconditioner * MLPrec = 
    new ML_Epetra::MultiLevelPreconditioner(*A, true);

  // Tell AztecOO to use this preconditioner.
  solver.SetPrecOperator(MLPrec);

  // Tell AztecOO to use CG to solve the problem.
  solver.SetAztecOption(AZ_solver, AZ_cg);

  // Tell AztecOO to output status information every iteration 
  // (hence the 1, which is the output frequency in terms of 
  // number of iterations).
  solver.SetAztecOption(AZ_output, 1);

  // Maximum number of iterations to try.
  int Niters = 150; 
  // Convergence tolerance.
  double tol = 1e-10;

  // Solve the linear problem.
  solver.Iterate (Niters, tol);

  // Print out some information about the preconditioner
  if (Comm.MyPID() == 0) 
    std::cout << MLPrec->GetOutputList();

  // We're done with the preconditioner now, so we can deallocate it.
  delete MLPrec;

  // Verify the solution by computing the residual explicitly.
  double residual = 0.0;
  double diff = 0.0;
  Gallery.ComputeResidual (&residual);
  Gallery.ComputeDiffBetweenStartingAndExactSolutions (&diff);

  // The Epetra_Time object has been keeping track of elapsed time
  // locally (on this MPI process).  Take the min and max globally
  // to find the min and max elapsed time over all MPI processes.
  double myElapsedTime = Time.ElapsedTime ();
  double minElapsedTime = 0.0;
  double maxElapsedTime = 0.0;
  (void) Comm.MinAll (&myElapsedTime, &minElapsedTime, 1);
  (void) Comm.MaxAll (&myElapsedTime, &maxElapsedTime, 1);

  if (Comm.MyPID()==0) {
    const int numProcs = Comm.NumProc ();
    std::cout << "||b-Ax||_2 = " << residual << std::endl
         << "||x_exact - x||_2 = " << diff << std::endl
         << "Min total time (s) over " << numProcs << " processes: " 
         << minElapsedTime << std::endl
         << "Max total time (s) over " << numProcs << " processes: "
         << maxElapsedTime << std::endl;
  }

#ifdef EPETRA_MPI
  MPI_Finalize() ;
#endif
  return(EXIT_SUCCESS);
}


