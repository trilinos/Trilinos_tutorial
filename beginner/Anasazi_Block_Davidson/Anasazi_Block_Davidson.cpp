#include "Epetra_CrsMatrix.h"
#include "Epetra_MultiVector.h"
#include "Galeri_Maps.h"
#include "Galeri_CrsMatrices.h"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "AnasaziBasicEigenproblem.hpp"
#include "AnasaziBlockDavidsonSolMgr.hpp"
#include "AnasaziEpetraAdapter.hpp"
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include "../../aprepro_vhelp.h"

int main(int argc, char *argv[])
{

#ifdef HAVE_MPI
  MPI_Init(&argc,&argv);
  Epetra_MpiComm Comm( MPI_COMM_WORLD );
#else
  Epetra_SerialComm Comm;
#endif

  Teuchos::ParameterList GaleriList;

  // The problem is defined on a 2D grid, global size is nx * nx.
  int nx = 30; 
  GaleriList.set("n", nx * nx);
  GaleriList.set("nx", nx);
  GaleriList.set("ny", nx);
  Teuchos::RCP<Epetra_Map> Map = Teuchos::rcp( Galeri::CreateMap("Linear", Comm, GaleriList) );
  Teuchos::RCP<Epetra_RowMatrix> A = Teuchos::rcp( Galeri::CreateCrsMatrix("Laplace2D", &*Map, GaleriList) );

  //  Variables used for the Block Davidson Method
  const int    nev         = 4;
  const int    blockSize   = 5;
  const int    numBlocks   = 8;
  const int    maxRestarts = 100;
  const double tol         = 1.0e-8;

  typedef Epetra_MultiVector MV;
  typedef Epetra_Operator OP;
  typedef Anasazi::MultiVecTraits<double, Epetra_MultiVector> MVT;

  // Create an Epetra_MultiVector for an initial vector to start the solver.
  // Note:  This needs to have the same number of columns as the blocksize.
  //
  Teuchos::RCP<Epetra_MultiVector> ivec = Teuchos::rcp( new Epetra_MultiVector(*Map, blockSize) );
  ivec->Random();

  // Create the eigenproblem.
  Teuchos::RCP<Anasazi::BasicEigenproblem<double, MV, OP> > problem =
    Teuchos::rcp( new Anasazi::BasicEigenproblem<double, MV, OP>(A, ivec) );

  // Inform the eigenproblem that the operator A is symmetric
  problem->setHermitian(true);

  // Set the number of eigenvalues requested
  problem->setNEV( nev );

  // Inform the eigenproblem that you are finishing passing it information
  bool boolret = problem->setProblem();
  if (boolret != true) {
    std::cout<<"Anasazi::BasicEigenproblem::setProblem() returned an error." << std::endl;
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  // Create parameter list to pass into the solver manager
  Teuchos::ParameterList anasaziPL;
  anasaziPL.set( "Which", "LM" );
  anasaziPL.set( "Block Size", blockSize );
  anasaziPL.set( "Num Blocks", numBlocks );
  anasaziPL.set( "Maximum Restarts", maxRestarts );
  anasaziPL.set( "Convergence Tolerance", tol );
  anasaziPL.set( "Verbosity", Anasazi::Errors+Anasazi::Warnings+Anasazi::TimingDetails+Anasazi::FinalSummary);

  // Create the solver manager
  Anasazi::BlockDavidsonSolMgr<double, MV, OP> anasaziSolver(problem, anasaziPL);

  // Solve the problem
  Anasazi::ReturnType returnCode = anasaziSolver.solve();

  // Get the eigenvalues and eigenvectors from the eigenproblem
  Anasazi::Eigensolution<double,MV> sol = problem->getSolution();
  std::vector<Anasazi::Value<double> > evals = sol.Evals;
  Teuchos::RCP<MV> evecs = sol.Evecs;

  // Compute residuals.
  std::vector<double> normR(sol.numVecs);
  if (sol.numVecs > 0) {
    Teuchos::SerialDenseMatrix<int,double> T(sol.numVecs, sol.numVecs);
    Epetra_MultiVector tempAevec( *Map, sol.numVecs );
    T.putScalar(0.0); 
    for (int i=0; i<sol.numVecs; i++) {
      T(i,i) = evals[i].realpart;
    }
    A->Apply( *evecs, tempAevec );
    MVT::MvTimesMatAddMv( -1.0, *evecs, T, 1.0, tempAevec );
    MVT::MvNorm( tempAevec, normR );
  }

  // Print the results
  std::cout<<"Solver manager returned " << (returnCode == Anasazi::Converged ? "converged." : "unconverged.") << std::endl;
  std::cout<<std::endl;
  std::cout<<"------------------------------------------------------"<<std::endl;
  std::cout<<std::setw(16)<<"Eigenvalue"
           <<std::setw(18)<<"Direct Residual"
           <<std::endl;
  std::cout<<"------------------------------------------------------"<<std::endl;
  for (int i=0; i<sol.numVecs; i++) {
    std::cout<<std::setw(16)<<evals[i].realpart
             <<std::setw(18)<<normR[i]/evals[i].realpart
             <<std::endl;
  }
  std::cout<<"------------------------------------------------------"<<std::endl;

#ifdef HAVE_MPI
  MPI_Finalize() ; 
#endif

  return 0;
}
