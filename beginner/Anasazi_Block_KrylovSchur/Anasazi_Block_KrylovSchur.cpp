// This example shows how to use the block Krylov-Schur method to
// compute a few of the largest singular values (sigma) and
// corresponding right singular vectors (v) for the matrix A by
// solving the symmetric problem:
//
//                  (A^T * (A * v) = sigma * v
//
// where A is an m by n real matrix that is derived from the simplest
// finite difference discretization of the 2-dimensional kernel
// K(s,t)dt, where
//
//                  K(s,t) = s(t-1)   if 0 <= s <= t <= 1
//                           t(s-1)   if 0 <= t <= s <= 1
//
// NOTE:  This example came from the ARPACK SVD driver dsvd.f
//
// The main solver parameters are the number of singular values to
// compute (numSingularValues), and the number of starting vectors
// (blockSize).  The implementation of getParameterList() below
// includes other parameters, like the iteration tolerance and the
// maximum number of restart cycles.
//
#include "AnasaziBlockKrylovSchurSolMgr.hpp"
#include "AnasaziBasicEigenproblem.hpp"
#include "AnasaziConfigDefs.hpp"
#include "AnasaziEpetraAdapter.hpp"

#include "Epetra_CrsMatrix.h"

// Include the appropriate communicator include files based on whether
// or not Trilinos was built with MPI.
#ifdef EPETRA_MPI
#  include "Epetra_MpiComm.h"
#  include <mpi.h>
#else
#  include "Epetra_SerialComm.h"
#endif
#include "Epetra_Map.h"

#include "../../aprepro_vhelp.h"
//
// Use Block Krylov-Schur iteration on A^T (A x) = (sigma)^2 x to
// compute the approximate SVD of A.
//
// Inputs:
//
// A: The operator A for which to compute the SVD.
// startingVector: One or more starting vectors for the iteration.
// numSingularValues: Number of singular values to compute.
//
// Outputs:
//
// 1. Approximate eigenvalues of A^T A.
// 2. Approximate eigenvectors of A^T A.
//
// Anasazi stores the eigenvalues of a real matrix as pairs of real
// values, using the Value struct.  Our operator A^T * A is symmetric,
// so the eigenvalues should all be real.  However, storing
// eigenvalues as (real, imaginary) pairs allows us to solve
// nonsymmetric problems as well.
//
std::pair< std::vector<Anasazi::Value<double> >, Teuchos::RCP<Epetra_MultiVector> >
solve (const Teuchos::RCP<Epetra_Operator>& A, 
       const Teuchos::RCP<Epetra_MultiVector>& startingVector,
       const int numSingularValues);

//
// Return a read-and-write view of V as an Anasazi::MultiVec<double>.
// We use this method in the solve() routine below.
//
Teuchos::RCP<Anasazi::MultiVec<double> >
createMultiVectorView (const Teuchos::RCP<Epetra_MultiVector>& V);

//
// Return a list of parameters to pass into Anasazi's Block
// Krylov-Schur solver.
//
// blockSize: Desired block size (number of starting vectors).
//
Teuchos::RCP<Teuchos::ParameterList> 
getParameterList (const int blockSize);

//
// Build an m by n (nonsquare) sparse matrix with entries
//
//          A(i,j) = k*(si)*(tj - 1) if i <= j
//                 = k*(tj)*(si - 1) if i  > j
//
// where si = i/(m+1) and tj = j/(n+1) and k = 1/(n+1).  We use this
// matrix to exercise Anasazi's Block Krylov-Schur solver.
//
Teuchos::RCP<Epetra_CrsMatrix> 
buildSparseMatrix (const Epetra_Comm& Comm,
                   const Epetra_Map& RowMap,
                   const Epetra_Map& ColMap,
                   const int m, 
                   const int n);

//
// The "main" driver routine.
//
int 
main (int argc, char *argv[]) 
{
  // These "using" statements make the code a bit more concise.
  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

  // Shorthand for a collection of operations on the Scalar=double type.
  typedef Teuchos::ScalarTraits<double> STS;
  // Shorthand for a collection of operations on Epetra vectors.
  typedef Anasazi::MultiVecTraits<double, Epetra_MultiVector> MVT;
  // Shorthand for a collection of operations on Epetra operators.
  typedef Anasazi::OperatorTraits<double, Epetra_MultiVector, Epetra_Operator> OPT;

  // If Trilinos was built with MPI, initialize MPI, otherwise
  // initialize the serial "communicator" that stands in for MPI.
#ifdef EPETRA_MPI
  MPI_Init (&argc,&argv);
  Epetra_MpiComm Comm (MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif

  // Number of singular values to compute.
  const int numSingularValues = 4; 

  // The number of columns in the starting (multi)vector.
  const int blockSize = 1; 

  // Dimensions of the matrix A: m rows and n columns.
  int m = 500;
  int n = 100;

  // Construct a Map that puts approximately the same number of rows
  // of the matrix A on each processor.
  Epetra_Map RowMap (m, 0, Comm);
  Epetra_Map ColMap (n, 0, Comm);

  // Create an Epetra_CrsMatrix using the above row and column maps,
  // and the given matrix dimensions m and n.
  RCP<Epetra_CrsMatrix> A = buildSparseMatrix (Comm, RowMap, ColMap, m, n);

  // Create a vector to be the solver's starting vector.
  //
  // The starting vector must have the same number of columns as the
  // "Block Size" parameter's value.  In this case, the initial vector
  // is in the domain of the matrix A, so it uses A's domain map as
  // its map.
  RCP<Epetra_MultiVector> initVec = 
    rcp (new Epetra_MultiVector (A->OperatorDomainMap(), blockSize));

  // Fill the initial vector with random data.
  MVT::MvRandom (*initVec);

  // Compute the SVD by solving A^T A x = (sigma)^2 x as an eigenvalue
  // problem.  Return the eigenvalues of A^T A, and the eigenvectors
  // of A^T A.  The solve() method is declared above and defined below
  // the definition of main(), in this file; it is a simple wrapper of
  // Anasazi's solver.
  std::pair< std::vector<Anasazi::Value<double> >, RCP<Epetra_MultiVector> > result = 
    solve (A, initVec, numSingularValues);
  std::vector<Anasazi::Value<double> >& evals = result.first;

  const int computedNumSingularValues = evals.size();
  if (computedNumSingularValues > 0) 
    {
      const double one = 1.0;
      const double zero = 0.0;
      // My (MPI) process rank.
      int MyPID = Comm.MyPID();

      //////////////////////////////////////////////////////////////////////
      //
      // Compute the singular values, singular vectors, and direct residuals.
      //
      //////////////////////////////////////////////////////////////////////    

      // The singular values of the matrix A are the square roots of the
      // eigenvalues of A^T * A.
      if (MyPID == 0) {
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << "Computed Singular Values: " << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
      }
      for (int i = 0; i < computedNumSingularValues; ++i) {
        // The operator A^T * A is symmetric, so the eigenvalues
        // should all have zero imaginary parts.
        evals[i].realpart = STS::squareroot (evals[i].realpart); 
      }

      //////////////////////////////////////////////////////////////////////
      // Compute the left singular vectors: u_j = (Av)_j / sigma_j
      //////////////////////////////////////////////////////////////////////

      std::vector<double> tempnrm (computedNumSingularValues);
      std::vector<double> directnrm (computedNumSingularValues);

      // The multivectors Av and u are in the range of the operator A,
      // so we have to create them using the range map of A.
      Epetra_MultiVector Av (A->OperatorRangeMap(), computedNumSingularValues);
      Epetra_MultiVector u (A->OperatorRangeMap(), computedNumSingularValues);

      // Compute Av = A*evecs and the norms of the columns of Av.
      RCP<Epetra_MultiVector> evecs = result.second;
      OPT::Apply (*A, *evecs, Av);
      MVT::MvNorm (Av, tempnrm);

      Teuchos::SerialDenseMatrix<int,double> S (computedNumSingularValues, 
                                                computedNumSingularValues);
      for (int i = 0; i < computedNumSingularValues; ++i) { 
        S(i,i) = one / tempnrm[i]; 
      }
      // u := Av * S + zero * u
      MVT::MvTimesMatAddMv (one, Av, S, zero, u);

      //////////////////////////////////////////////////////////////////////
      // Compute direct residuals: || (Av - sigma*u)_j ||_2
      //////////////////////////////////////////////////////////////////////

      for (int i = 0; i < computedNumSingularValues; ++i) { 
        S(i,i) = evals[i].realpart; 
      }
      // Av := -one * u * S + one * Av
      //     = Av - u*S.
      MVT::MvTimesMatAddMv (-one, u, S, one, Av);

      // directnrm_j = || (Av)_j ||_2
      MVT::MvNorm (Av, directnrm);

      //////////////////////////////////////////////////////////////////////
      // Print results to stdout on MPI process 0.
      //////////////////////////////////////////////////////////////////////

      if (MyPID == 0) {
        // It's rude to set the ostream flags without restoring their 
        // original values when you're done.
        std::ios_base::fmtflags originalFlags = std::cout.flags ();

        // Set the flags on cout for nice neat output.
        std::cout.setf (std::ios_base::right, std::ios_base::adjustfield);
        std::cout << std::setw(16) << "Singular Value"
             << std::setw(20) << "Direct Residual"
             << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
        for (int i = 0; i < computedNumSingularValues; ++i) {
          std::cout << std::setw(16) << evals[i].realpart
               << std::setw(20) << directnrm[i] 
               << std::endl;
        }  
        std::cout << "------------------------------------------------------" << std::endl;

        // Restore cout's original flags.
        std::cout.flags (originalFlags);
      }
    }

#ifdef EPETRA_MPI
  MPI_Finalize() ;
#endif
  return 0;
}


std::pair< std::vector<Anasazi::Value<double> >, Teuchos::RCP<Epetra_MultiVector> >
solve (const Teuchos::RCP<Epetra_Operator>& A, 
       const Teuchos::RCP<Epetra_MultiVector>& startingVector,
       const int numSingularValues)
{
  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;
  
  const Epetra_Comm& Comm = A->Comm();

  // Set the data type of the matrix operator and multivectors.
  //
  // Anasazi solvers work with many different operator and multivector
  // types.  In general, you may use Epetra, Tpetra, or Thyra types
  // directly, or even your own type.  If you use your own MV and OP
  // types, you must either make them inherit from Anasazi::MultiVec
  // resp. Anasazi::Operator, or specialize the
  // Anasazi::MultiVecTraits and Anasazi::OperatorTraits traits
  // classes for your MV and OP types.
  //
  // We choose MV and OP as we do below because of how we are solving
  // the SVD.  Since we are solving it via A^T (A x) = (sigma)^2 x, we
  // need to construct an operator that uses A to apply A^T * A.  The
  // new operator is of type Anasazi::EpetraSymOp, which is-an
  // Anasazi::Operator<double>.  As a result, in order to use the new
  // operator in an Anasazi solver, we must set the template
  // parameters to MV = Anasazi::MultiVec<double> and OP =
  // Anasazi::Operator<double>.
  typedef Anasazi::MultiVec<double> MV;
  typedef Anasazi::Operator<double> OP;

  // The MultiVecTraits traits class maps from the Scalar and MV (see
  // above) types to a common set of operations on MV objects.  It is
  // the preferred public interface of all MV objects that Anasazi
  // accepts.  There is a corresponding OperatorTraits traits class
  // for OP objects as well, that provides a minimal common set of
  // operations for OP objects.
  typedef Anasazi::MultiVecTraits<double, MV> MVT;
  typedef Anasazi::OperatorTraits<double, MV, OP> OPT;

  // Get solver parameters for Block Krylov-Schur.
  RCP<ParameterList> plist = getParameterList (startingVector->NumVectors());

  // Create the operator that uses the matrix A to represent the A^T *
  // A operator.  ATA does not copy A; it just keeps a pointer.
  RCP<OP> ATA = rcp (new Anasazi::EpetraSymOp (A));

  // As mentioned above, we must use MV = Anasazi::MultiVec<double>.
  // However, our starting vector is an Epetra_MultiVector, not an
  // Anasazi::MultiVec<double>.  Thus, we have to wrap it in an
  // Anasazi::EpetraMultiVec, which is-an Anasazi::MultiVec<double>.
  RCP<MV> initVec = createMultiVectorView (startingVector);

  // Create the object that holds the eigenvalue problem to solve.
  // The problem object is templated on the Scalar, MultiVector (MV),
  // and Operator (OP) types.
  RCP<Anasazi::BasicEigenproblem<double, MV, OP> > problem =
    rcp (new Anasazi::BasicEigenproblem<double, MV, OP> (ATA, initVec));

  // Inform the eigenproblem that the operator (A^T * A) is symmetric.
  problem->setHermitian (true);

  // Set the number of eigenvalues (singular values, in this case) to
  // compute.
  problem->setNEV (numSingularValues);

  // Inform the eigenvalue problem that you are finished passing it
  // information.
  TEUCHOS_TEST_FOR_EXCEPTION( ! problem->setProblem(), 
                      std::runtime_error,
                      "Failed to set the eigenvalue problem." );

  // Initialize the Block Krylov-Schur solver.  The solver may fill in
  // the given parameter list with defaults for any parameters that
  // were not supplied.  Thus, you don't have to know all the
  // parameters, just those that matter to you.
  Anasazi::BlockKrylovSchurSolMgr<double, MV, OP> MySolverMgr (problem, *plist);

  // Solve the problem to the specified tolerance or number of iterations.
  Anasazi::ReturnType returnCode = MySolverMgr.solve();

  if (returnCode != Anasazi::Converged && Comm.MyPID() == 0) {
    std::cout << "The Anasazi solver's solve() routine returned Unconverged." 
         << std::endl;
  }

  // Get the eigenvalues and eigenvectors from the eigenproblem.
  Anasazi::Eigensolution<double, MV> sol = problem->getSolution();

  // sol.Evecs is-an Anasazi::EpetraMultiVec, which in turn is-an
  // Epetra_MultiVector.  Cast it to Epetra_MultiVector so that the
  // calling code knows how to deal with it.
  using Teuchos::rcp_dynamic_cast;
  RCP<Epetra_MultiVector> eigenvecs = 
    rcp_dynamic_cast<Anasazi::EpetraMultiVec> (sol.Evecs);

  // Return the eigenvalues and eigenvectors.  
  return std::make_pair (sol.Evals, eigenvecs);
}


Teuchos::RCP<Anasazi::MultiVec<double> >
createMultiVectorView (const Teuchos::RCP<Epetra_MultiVector>& V)
{
  using Teuchos::rcp;
  typedef Anasazi::MultiVecTraits<double, Epetra_MultiVector> MVT;
  const int numVecs = MVT::GetNumberVecs (*V);

  // Creating a view of V requires telling Anasazi the indices of the
  // columns of V to view.
  //
  // The current (10.6) release of Trilinos refers to a range of
  // columns of a multivector using an std::vector of their
  // (zero-based) column indices.  The development branch, and the
  // next release, will also allow using a Teuchos::Range1D to refer
  // to a range of columns of a multivector.
  std::vector<int> index (numVecs);
  for (int j = 0; j < numVecs; ++j)
    index[j] = j;

  // Anasazi::EpetraMultiVec is-an Anasazi::MultiVec<double>.
  //
  // NOTE: If V is deleted, the view will no longer be valid.  Epetra
  // views are not safe in that respect.  Tpetra views _are_ safe
  // because the view holds an RCP to the original vector, preventing
  // it from being deleted until after the view is deleted.
  return rcp (new Anasazi::EpetraMultiVec (View, *V, index));
}


Teuchos::RCP<Epetra_CrsMatrix> 
buildSparseMatrix (const Epetra_Comm& Comm,
                   const Epetra_Map& RowMap,
                   const Epetra_Map& ColMap,
                   const int m, 
                   const int n)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  typedef Teuchos::ScalarTraits<double> STS;

  const double one = STS::one();
  const double zero = STS::zero();

  // My MPI process rank.
  const int MyPID = Comm.MyPID();

  // Get update list and number of local equations from newly created Map.
  const int NumMyRowElements = RowMap.NumMyElements ();
  std::vector<int> MyGlobalRowElements (NumMyRowElements);
  RowMap.MyGlobalElements (&MyGlobalRowElements[0]);

  // Create an Epetra_CrsMatrix using the given row map.
  RCP<Epetra_CrsMatrix> A = rcp (new Epetra_CrsMatrix (Copy, RowMap, n));

  // We use info to catch any errors that may have happened during
  // matrix assembly, and report them globally.  We do this so that
  // the MPI processes won't call FillComplete() unless they all
  // successfully filled their parts of the matrix.
  int info = 0;
  try {
    //
    // Compute coefficients for the discrete integral operator.
    //
    std::vector<double> Values (n);
    std::vector<int> Indices (n);
    const double inv_mp1 = one / (m+1);
    const double inv_np1 = one / (n+1);
    for (int i = 0; i < n; ++i) { 
      Indices[i] = i; 
    }
    for (int i = 0; i < NumMyRowElements; ++i) {
      for (int j = 0; j < n; ++j) 
        {
          if (MyGlobalRowElements[i] <= j) {
            Values[j] = inv_np1 * 
              ( (MyGlobalRowElements[i]+one)*inv_mp1 ) * 
              ( (j+one)*inv_np1 - one );  // k*(si)*(tj-1)
          }
          else {
            Values[j] = inv_np1 * 
              ( (j+one)*inv_np1 ) * 
              ( (MyGlobalRowElements[i]+one)*inv_mp1 - one );  // k*(tj)*(si-1)
          }
        }
      info = A->InsertGlobalValues (MyGlobalRowElements[i], n, 
                                    &Values[0], &Indices[0]);
      // Make sure that the insertion succeeded.  Teuchos'
      // TEUCHOS_TEST_FOR_EXCEPTION macro gives a nice error message if the
      // thrown exception isn't caught.  We'll report this on the
      // offending MPI process.
      TEUCHOS_TEST_FOR_EXCEPTION( info != 0, std::runtime_error, "Failed to insert n=" 
                          << n << " global value" << (n != 1 ? "s" : "") 
                          << " in row " << MyGlobalRowElements[i] 
                          << " of the matrix." );
    } // for i = 0...
    
    // Call FillComplete on the matrix.  Since the matrix isn't square,
    // we have to give FillComplete the domain and range maps, which in
    // this case are the column resp. row maps.
    info = A->FillComplete (ColMap, RowMap);
    TEUCHOS_TEST_FOR_EXCEPTION( info != 0, std::runtime_error, 
                        "FillComplete failed with INFO = " << info << ".");
    info = A->OptimizeStorage();
    TEUCHOS_TEST_FOR_EXCEPTION( info != 0, std::runtime_error, 
                        "OptimizeStorage failed with INFO = " << info << ".");
  } catch (std::runtime_error& e) {
    // If multiple MPI processes are reporting errors, sometimes
    // forming the error message as a string and then writing it to
    // the output stream prevents messages from different processes
    // from being interleaved.
    std::ostringstream os;
    os << "*** Error on MPI process " << MyPID << ": " << e.what();
    std::cerr << os.str() << std::endl;
    if (info == 0)
      info = -1; // All procs will share info later on.
  }

  // Do a reduction on the value of info, to ensure that all the MPI
  // processes successfully filled the sparse matrix.
  {
    int minInfo = 0;
    int maxInfo = 0;

    // Test both info < 0 and info > 0.
    Comm.MinAll (&info, &minInfo, 1);
    Comm.MaxAll (&info, &maxInfo, 1);
    TEUCHOS_TEST_FOR_EXCEPTION( minInfo != 0 || maxInfo != 0, std::runtime_error,
                        "Filling and assembling the sparse matrix failed." );
  }
  
  
  // Shut down Epetra Warning tracebacks.
  A->SetTracebackMode (1);

  return A;
}


Teuchos::RCP<Teuchos::ParameterList> 
getParameterList (const int blockSize)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;

  RCP<ParameterList> plist = rcp (new ParameterList ("Block Krylov-Schur"));

  // "Num Blocks" parameter is the Krylov basis length to use.
  const int numBlocks = 10;

  // "Maximum Restarts" parameter is the maximum number of times to
  // restart the Krylov methods.
  const int maxRestarts = 20;

  // For this test, we set the "Convergence Tolerance" parameter to
  // machine precision for double-precision floating-point values.
  // Teuchos' ScalarTraits traits class knows what machine precision
  // is for many different floating-point types.  We could also call
  // LAPACK's DLAMCH('E') to get machine precision.
  const double tol = Teuchos::ScalarTraits<double>::eps();

  // The "Which" parameter governs the order in which Anasazi computes
  // eigenvalues (or singular values, in this case).  "LM" means the
  // eigenvalues of Largest Magnitude.  Valid options are "SM", "LM",
  // "SR", and "LR".  These mean, respectively, "Smallest Magnitude,"
  // "Largest Magnitude," "Smallest Real", and "Largest Real."  
  //
  // These abbreviations are exactly the same as those used by
  // ARPACK's "which" parameter, and in turn by Matlab's "eigs"
  // function (which calls ARPACK internally).
  const std::string which = "LM";

  plist->set ("Which", which);
  plist->set ("Block Size", blockSize);
  plist->set ("Num Blocks", numBlocks);
  plist->set ("Maximum Restarts", maxRestarts);
  plist->set ("Convergence Tolerance", tol);
  // Tell Anasazi to print output for errors, warnings, timing
  // details, and the final summary of results.
  plist->set ("Verbosity", Anasazi::Errors + Anasazi::Warnings + 
              Anasazi::TimingDetails + Anasazi::FinalSummary);

  return plist;
}


