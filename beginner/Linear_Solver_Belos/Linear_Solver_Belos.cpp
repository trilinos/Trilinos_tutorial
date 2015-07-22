#include "Epetra_CrsMatrix.h"
#include "Epetra_MultiVector.h"
#include "Epetra_LinearProblem.h"
#include "Galeri_Maps.h"
#include "Galeri_CrsMatrices.h"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Ifpack.h"
#include "Ifpack_AdditiveSchwarz.h"
#include "BelosLinearProblem.hpp"
#include "BelosBlockGmresSolMgr.hpp"
#include "BelosEpetraAdapter.hpp"
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include "../../aprepro_vhelp.h"

int
main (int argc, char *argv[])
{
  using std::cout;
  using std::endl;
  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

#ifdef HAVE_MPI
  MPI_Init(&argc,&argv);
  Epetra_MpiComm Comm( MPI_COMM_WORLD );
#else
  Epetra_SerialComm Comm;
#endif
  const int myRank = Comm.MyPID();

  ParameterList GaleriList;

  // The problem is defined on a 2D grid, global size is nx * nx.
  int nx = 30; 
  GaleriList.set("n", nx * nx);
  GaleriList.set("nx", nx);
  GaleriList.set("ny", nx);
  RCP<Epetra_Map> Map = 
    rcp (Galeri::CreateMap("Linear", Comm, GaleriList));
  // "&*Map" turns an RCP<Map> into a raw pointer, which is what
  // Galeri::CreateCrsMatrix() wants.
  RCP<Epetra_RowMatrix> A = 
    rcp (Galeri::CreateCrsMatrix("Laplace2D", &*Map, GaleriList));
  TEUCHOS_TEST_FOR_EXCEPTION(A == Teuchos::null, std::runtime_error,
                     "Galeri returned a null operator A.");

  // =============================================================== //
  // B E G I N N I N G   O F   I F P A C K   C O N S T R U C T I O N //
  // =============================================================== //

  ParameterList List;

  // Allocate an IFPACK factory.  The object contains no data, only
  // the Create() method for creating preconditioners.
  Ifpack Factory;

  // Create the preconditioner.  For the list of PrecType values that
  // Create() accepts, please check the IFPACK documentation.
  std::string PrecType = "ILU"; // incomplete LU
  int OverlapLevel = 1; // must be >= 0. If Comm.NumProc() == 1,
                        // it is ignored.

  RCP<Ifpack_Preconditioner> Prec = 
    rcp (Factory.Create (PrecType, &*A, OverlapLevel));
  TEUCHOS_TEST_FOR_EXCEPTION(Prec == Teuchos::null, std::runtime_error,
                     "IFPACK failed to create a preconditioner of type \"" 
                     << PrecType << "\" with overlap level " 
                     << OverlapLevel << ".");

  // Specify parameters for ILU.  ILU is local to each MPI process.
  List.set("fact: drop tolerance", 1e-9);
  List.set("fact: level-of-fill", 1);

  // IFPACK uses overlapping Schwarz domain decomposition over all
  // participating processes to combine the results of ILU on each
  // process.  IFPACK's Schwarz method can use any of the following
  // combine modes to combine overlapping results:
  //
  // "Add", "Zero", "Insert", "InsertAdd", "Average", "AbsMax"
  //
  // The Epetra_CombineMode.h header file defines their meaning.
  List.set("schwarz: combine mode", "Add");
  // Set the parameters.
  IFPACK_CHK_ERR(Prec->SetParameters(List));

  // Initialize the preconditioner. At this point the matrix must have
  // been FillComplete()'d, but actual values are ignored.
  IFPACK_CHK_ERR(Prec->Initialize());

  // Build the preconditioner, by looking at the values of the matrix.
  IFPACK_CHK_ERR(Prec->Compute());

  // Create the Belos preconditioned operator from the Ifpack preconditioner.
  // NOTE:  This is necessary because Belos expects an operator to apply the
  //        preconditioner with Apply() NOT ApplyInverse().
  RCP<Belos::EpetraPrecOp> belosPrec = rcp (new Belos::EpetraPrecOp (Prec));

  // =================================================== //
  // E N D   O F   I F P A C K   C O N S T R U C T I O N //
  // =================================================== //

  // At this point, we need some additional objects
  // to define and solve the linear system.

  // Define the left-hand side (the solution / initial guess vector)
  // and right-hand side.  The solution is in the domain of the
  // operator A, and the right-hand side is in the range of A. 
  RCP<Epetra_MultiVector> LHS = rcp (new Epetra_MultiVector (A->OperatorDomainMap (), 1));
  RCP<Epetra_MultiVector> RHS = rcp (new Epetra_MultiVector (A->OperatorDomainMap (), 1));

  // Make the exact solution a vector of all ones.
  LHS->PutScalar(1.0);
  // Compute RHS := A * LHS.
  A->Apply(*LHS,*RHS);

  // Now randomize the right-hand side.
  RHS->Random();

  // Need a Belos::LinearProblem to define a Belos solver   
  typedef Epetra_MultiVector                MV;
  typedef Epetra_Operator                   OP;    
  RCP<Belos::LinearProblem<double,MV,OP> > problem
    = rcp (new Belos::LinearProblem<double,MV,OP>(A, LHS, RHS));

  // Set the IFPACK preconditioner.
  //
  // We're using it as a right preconditioner.  It's better to use a
  // right preconditioner than a left preconditioner in GMRES, because
  // otherwise the projected problem will have a different residual
  // (in exact arithmetic) than the original problem.  This makes it
  // harder for GMRES to tell when it has converged.
  problem->setRightPrec (belosPrec);

  bool set = problem->setProblem();
  TEUCHOS_TEST_FOR_EXCEPTION( ! set, 
                      std::runtime_error, 
                      "*** Belos::LinearProblem failed to set up correctly! ***");

  // Create a parameter list to define the Belos solver.
  RCP<ParameterList> belosList = rcp (new ParameterList ());
  belosList->set ("Block Size", 1);              // Blocksize to be used by iterative solver
  belosList->set ("Maximum Iterations", 1550);   // Maximum number of iterations allowed
  belosList->set ("Convergence Tolerance", 1e-8);// Relative convergence tolerance requested
  belosList->set ("Verbosity", Belos::Errors+Belos::Warnings+Belos::TimingDetails+Belos::FinalSummary );

  // Create an iterative solver manager.
  Belos::BlockGmresSolMgr<double,MV,OP> belosSolver (problem, belosList);

  // Perform solve.
  Belos::ReturnType ret = belosSolver.solve();

  // Did we converge?
  if (myRank == 0) {
    if (ret == Belos::Converged) {
      std::cout << "Belos converged." << std::endl;
    } else {
      std::cout << "Belos did not converge." << std::endl;
    }
  }

  // Print out the preconditioner.  IFPACK preconditioner objects know
  // how to print themselves in parallel directly to std::cout.
  std::cout << *Prec;

#ifdef HAVE_MPI
  MPI_Finalize() ; 
#endif
  return 0;
}
