// Use of ML as a preconditioner.  Set some non-default options.

#include "Epetra_ConfigDefs.h"
#ifdef HAVE_MPI
#include "mpi.h"
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif
#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_LinearProblem.h"
#include "Epetra_Time.h"
#include "AztecOO.h"

#include "../../aprepro_vhelp.h"

#include "Teuchos_ParameterList.hpp"

// includes required by ML
#include "ml_epetra_preconditioner.h"

#include "Trilinos_Util_CrsMatrixGallery.h"

using namespace Teuchos;
using namespace Trilinos_Util;

#include <iostream>

int main(int argc, char *argv[])
{

#ifdef EPETRA_MPI
  MPI_Init(&argc,&argv);
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;

#endif

  Epetra_Time Time(Comm);

  // initialize an Gallery object
  CrsMatrixGallery Gallery("laplace_3d", Comm);

  Gallery.Set("problem_size", 1000);

  // retrieve pointers to matrix and linear problem
  Epetra_RowMatrix * A = Gallery.GetMatrix();

  Epetra_LinearProblem * Problem = Gallery.GetLinearProblem();

  // Construct a solver object for this problem
  AztecOO solver(*Problem);

  Teuchos::ParameterList MLList;
  //set multigrid defaults based on problem type
  //  SA is appropriate for Laplace-like systems
  //  NSSA is appropriate for nonsymmetric problems such as convection-diffusion
  ML_Epetra::SetDefaults("SA",MLList);

  // output level, 0 being silent and 10 verbose
  MLList.set("ML output", 10);

  // maximum number of levels possible
  MLList.set("max levels",5);

  //common smoother options: Chebyshev, Gauss-Seidel, symmetric Gauss-Seidel, Jacobi, ILU, IC
  MLList.set("smoother: type","Chebyshev");
  MLList.set("smoother: sweeps",2);

  //set a different smoother on the first coarse level (finest level = 0)
  MLList.set("smoother: type (level 1)","symmetric Gauss-Seidel");
  MLList.set("smoother: sweeps (level 1)",4);

  // use both pre and post smoothing
  MLList.set("smoother: pre or post", "both");

  //coarsest level solve.  One can use any smoother here, as well.
  MLList.set("coarse: type","Amesos-KLU");

  // coarsening options:  Uncoupled, MIS, Uncoupled-MIS (uncoupled on the finer grids, then switch to MIS)
  MLList.set("aggregation: type", "Uncoupled");

  // create the preconditioner object based on options in MLList and compute hierarchy
  ML_Epetra::MultiLevelPreconditioner * MLPrec = new ML_Epetra::MultiLevelPreconditioner(*A, MLList);

  // tell AztecOO to use this preconditioner, then solve
  solver.SetPrecOperator(MLPrec);

  solver.SetAztecOption(AZ_solver, AZ_cg);
  solver.SetAztecOption(AZ_output, 1);

  int Niters = 500;
  solver.Iterate(Niters, 1e-12);

  // print out some information about the preconditioner
  if( Comm.MyPID() == 0 ) std::cout << MLPrec->GetOutputList();

  delete MLPrec;

  // compute the real residual

  double residual, diff;

  Gallery.ComputeResidual(&residual);
  Gallery.ComputeDiffBetweenStartingAndExactSolutions(&diff);

  if( Comm.MyPID()==0 ) {

    std::cout << "||b-Ax||_2 = " << residual << std::endl;
    std::cout << "||x_exact - x||_2 = " << diff << std::endl;

    std::cout << "Total Time = " << Time.ElapsedTime() << std::endl;
  }

  if (residual > 1e-5)

    exit(EXIT_FAILURE);
#ifdef EPETRA_MPI
  MPI_Finalize() ;
#endif
  return(EXIT_SUCCESS);
}


