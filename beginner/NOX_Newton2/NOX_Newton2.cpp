//
// Simple example of solving the following nonlinear partial
// differential equation (PDE):
//
// -\Delta u + \lambda e^u = 0  in \Omega = (0,1) \times (0,1)
//                       u = 0  on \partial \Omega
//
// using NOX (Trilinos' Nonlinear Object-Oriented Solutions package).
// For more details and documentation, see the NOX web site:
//
// http://trilinos.sandia.gov/packages/nox/
//
#ifdef HAVE_MPI
#  include "mpi.h"
#  include "Epetra_MpiComm.h"
#else
#  include "Epetra_SerialComm.h"
#endif
#include "Epetra_Map.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_Vector.h"

#include "NOX.H"
#include "NOX_Epetra_Interface_Required.H"
#include "NOX_Epetra_Interface_Jacobian.H"
#include "NOX_Epetra_LinearSystem_AztecOO.H"
#include "NOX_Epetra_Group.H"

#include "../../aprepro_vhelp.h"

//
// Report the the number of lower, upper, left and right nodes, for
// each element of a Cartesian grid containing nx x ny elements.
//
static void  
getNeighbors (const int i, const int nx, const int ny,
              int & left, int & right, 
              int & lower, int & upper) 
{
  int ix, iy;
  ix = i % nx;

  iy = (i - ix) / nx;

  if( ix == 0 ) 
    left = -1;
  else 
    left = i-1;

  if( ix == nx-1 ) 
    right = -1;
  else
    right = i+1;

  if( iy == 0 ) 
    lower = -1;
  else
    lower = i-nx;

  if( iy == ny-1 ) 
    upper = -1;
  else
    upper = i+nx;

  return;
}

//
// Return a new sparse matrix corresponding to the discretization of a
// Laplacian over a 2-D Cartesian grid, with nx grid points along the
// x axis and ny grid points along the y axis. For simplicity's sake,
// we assume that all nodes in the grid are internal (Dirichlet
// boundary nodes should alread have been condensated).
//
// Input arguments:
// 
// nx: Number of (internal) grid points along the x axis.
// ny: Number of (internal) grid points along the y axis.
// Comm: The communicator over which to distribute the returned sparse
//   matrix.
//
// Return: The sparse matrix, distributed over the given communicator.
//
Teuchos::RCP<Epetra_CrsMatrix>
CreateLaplacian (const int nx, 
                 const int ny,
                 const Epetra_Comm& Comm)
{
  using Teuchos::RCP;
  using Teuchos::rcp;

  int NumGlobalElements = nx * ny;

  // Create a map that distributes the elements evenly over the
  // process(es) participating in the given communicator.
  Epetra_Map Map (NumGlobalElements, 0, Comm);

  // Local number of rows
  int NumMyElements = Map.NumMyElements();

  // Get the list of the global elements that belong to my MPI process.
  int* MyGlobalElements = Map.MyGlobalElements();

  // Grid 
  double hx = 1.0/(nx-1);
  double hy = 1.0/(ny-1);

  double off_left  = -1.0/(hx*hx);
  double off_right = -1.0/(hx*hx);
  double off_lower = -1.0/(hy*hy);
  double off_upper = -1.0/(hy*hy);
  double diag      =  2.0/(hx*hx) + 2.0/(hy*hy);

  int left, right, lower, upper;

  // "5" is a conservative (slightly too large in some cases)
  // overestimate of the number of nonzero elements per row of the
  // sparse matrix.  The estimate doesn't have to be exact since we
  // are using Epetra_CrsMatrix in dynamic allocation mode.
  RCP<Epetra_CrsMatrix> A = rcp (new Epetra_CrsMatrix (Copy, Map, 5));

  // Add rows one at a time to the sparse matrix.
  std::vector<double> Values (4);
  std::vector<int> Indices (4);

  for (int i = 0; i < NumMyElements; ++i) 
    {
      int NumEntries = 0;
      getNeighbors (MyGlobalElements[i], nx, ny, 
                    left, right, lower, upper);
      if (left != -1) {
        Indices[NumEntries] = left;
        Values[NumEntries] = off_left;
        ++NumEntries;
      }
      if (right != -1) {
        Indices[NumEntries] = right;
        Values[NumEntries] = off_right;
        ++NumEntries;
      }
      if (lower != -1) {
        Indices[NumEntries] = lower;
        Values[NumEntries] = off_lower;
        ++NumEntries;
      }
      if (upper != -1) {
        Indices[NumEntries] = upper;
        Values[NumEntries] = off_upper;
        ++NumEntries;
      }
      // Insert the off-diagonal entries first.  This is not required
      // for CrsMatrix, but it is apparently an old habit for many
      // Aztec users.
      A->InsertGlobalValues (MyGlobalElements[i], NumEntries, &Values[0], &Indices[0]);

      // Insert the diagonal entry.
      A->InsertGlobalValues (MyGlobalElements[i], 1, &diag, MyGlobalElements+i);
    }

  // Let Epetra reorganize the matrix and compute the global ordering.
  A->FillComplete();

  return A;
}

// ==========================================================================
// The PDEProblem class defines the nonlinear problem to solve.
// It provides two methods:
//
// ComputeF: computes F(x) for a given Epetra_Vector x
//
// UpdateJacobian: updates the entries of the Jacobian matrix.  
//
// The Jacobian matrix in this case can be written as
//
//     J = L + diag(lambda*exp(x[i])),
//
// where L corresponds to the discretization of a Laplacian, and diag
// is a diagonal matrix with entries lambda*exp(x[i]).  Thus, to
// update the jacobian we simply update the diagonal entries, by
// supplying x as a vector.  Similarly, to compute F(x), we reset J to
// be equal to L, multiply it by the (distributed) vector x, and then
// add the diagonal contribution.
// ==========================================================================
class PDEProblem {
public:
  //
  // Constructor.
  //
  // Input arguments:
  //
  // nx: Number of (internal) elements along the x axis.
  // ny: Number of (internal) elements along the y axis.
  // lambda: Scaling parameter of the Jacobian matrix.
  // Comm: Communicator over which to distribute the matrix.
  //
  PDEProblem (const int nx, 
              const int ny, 
              const double lambda,
              const Epetra_Comm& Comm) :
    nx_(nx), ny_(ny), lambda_(lambda) 
  {

    hx_ = 1.0/(nx_-1);
    hy_ = 1.0/(ny_-1);
    Matrix_ = CreateLaplacian (nx_,ny_, Comm);
  }

  // The destructor doesn't need to do anything.  RCPs are smart
  // pointers; they handle deallocation automatically.
  ~PDEProblem() {}

  // Compute F(x).
  void 
  ComputeF (const Epetra_Vector& x, Epetra_Vector& f) 
  {
    // Reset the diagonal entries.
    double diag = 2.0/(hx_*hx_) + 2.0/(hy_*hy_);

    int NumMyElements = Matrix_->Map().NumMyElements();

    // Get the list of the global elements that belong to my MPI process.
    int* MyGlobalElements = Matrix_->Map ().MyGlobalElements ();

    for (int i = 0; i < NumMyElements; ++i) {
      // Update the diagonal entry of the matrix.
      Matrix_->ReplaceGlobalValues (MyGlobalElements[i], 1, &diag, MyGlobalElements+i);
    }
    // Sparse matrix-vector product.  
    // Interprocess communication happens here.
    Matrix_->Multiply (false, x, f);

    for (int i = 0; i < NumMyElements; ++i) {
      // Include the contribution from the current diagonal entry.
      f[i] += lambda_*exp(x[i]);
    }
  }

  // Update the Jacobian matrix for a given vector x (see class
  // documentation for an explanation of how x contributes to the
  // update formula).
  void 
  UpdateJacobian (const Epetra_Vector & x) 
  {
    double diag =  2.0/(hx_*hx_) + 2.0/(hy_*hy_);

    int NumMyElements = Matrix_->Map ().NumMyElements ();

    // Get the list of the global elements that belong to my MPI process.
    int* MyGlobalElements = Matrix_->Map ().MyGlobalElements ();

    for (int i = 0; i < NumMyElements; ++i) {
      // Update the current diagonal entry.
      double newdiag = diag + lambda_*exp(x[i]);
      Matrix_->ReplaceGlobalValues (MyGlobalElements[i], 1, 
                                    &newdiag, MyGlobalElements+i);
    }

  }

  // Return a pointer to the internally stored matrix.
  Teuchos::RCP<Epetra_CrsMatrix> GetMatrix() {
    return Matrix_;
  }

private:
  int nx_, ny_;
  double hx_, hy_;
  Teuchos::RCP<Epetra_CrsMatrix> Matrix_;
  double lambda_;
};

// ==========================================================================
// SimpleProblemInterface defines the interface between NOX and our
// nonlinear problem to solve.  Its constructor accepts a PDEProblem
// object, which it uses to update the Jacobian and compute F(x).
// This interface is a bit crude; for example, it does not provide a
// PrecMatrix or Preconditioner.
// ==========================================================================
class SimpleProblemInterface : 
  public NOX::Epetra::Interface::Required,
  public NOX::Epetra::Interface::Jacobian
{
public:

  // The constructor takes a PDEProblem pointer.
  SimpleProblemInterface (Teuchos::RCP<PDEProblem> Problem) :
    Problem_(Problem) 
  {}

  // The destructor doesn't need to do anything, because RCPs are
  // smart pointers; they handle deallocation automatically.
  ~SimpleProblemInterface() {}

  bool 
  computeF (const Epetra_Vector& x, 
            Epetra_Vector& f,
            NOX::Epetra::Interface::Required::FillType F)
  {
    Problem_->ComputeF (x, f);
    return true;
  }

  bool 
  computeJacobian (const Epetra_Vector& x, Epetra_Operator& Jac)
  {
    Problem_->UpdateJacobian (x);
    return true;
  }

  bool 
  computePrecMatrix (const Epetra_Vector& x, Epetra_RowMatrix& M) 
  {
    throw std::runtime_error ("*** SimpleProblemInterface does not implement "
                              "computing an explicit preconditioner from an "
                              "Epetra_RowMatrix ***");
  }  

  bool 
  computePreconditioner(const Epetra_Vector & x, Epetra_Operator & O)
  {
    throw std::runtime_error ("*** SimpleProblemInterface does not implement "
                              "computing an explicit preconditioner from an "
                              "Epetra_Operator ***");
  }  

private:
  Teuchos::RCP<PDEProblem> Problem_;
};

//
// Test driver routine.
//
int 
main (int argc, char **argv)
{
  using Teuchos::ParameterList;
  using Teuchos::parameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif

  // Parameters for setting up the nonlinear PDE.  The 2-D regular
  // mesh on which the PDE's discretization is defined is nx by ny
  // (internal nodes; we assume Dirichlet boundary conditions have
  // been condensed out).
  const int nx = 5;
  const int ny = 6;  
  const double lambda = 1.0;

  RCP<PDEProblem> Problem = rcp (new PDEProblem (nx, ny, lambda, Comm));

  // Prepare the initial guess vector.  It should be a vector in the
  // domain of the nonlinear problem's matrix.
  Epetra_Vector InitialGuess (Problem->GetMatrix ()->OperatorDomainMap ());

  // Make the starting solution a zero vector.
  InitialGuess.PutScalar(0.0);

  // Set up the problem interface.
  RCP<SimpleProblemInterface> interface = 
    rcp (new SimpleProblemInterface (Problem));

  // Create the top-level parameter list to control NOX.
  //
  // "parameterList" (lowercase initial "p") is a "nonmember
  // constructor" that returns an RCP<ParameterList> with the
  // given name.
  RCP<ParameterList> params = parameterList ("NOX");

  // Tell the nonlinear solver to use line search.
  params->set ("Nonlinear Solver", "Line Search Based");

  //
  // Set the printing parameters in the "Printing" sublist.
  //
  ParameterList& printParams = params->sublist ("Printing");
  printParams.set ("MyPID", Comm.MyPID ()); 
  printParams.set ("Output Precision", 3);
  printParams.set ("Output Processor", 0);
  printParams.set ("Output Information", 
                   NOX::Utils::OuterIteration + 
                   NOX::Utils::OuterIterationStatusTest + 
                   NOX::Utils::InnerIteration +
                   NOX::Utils::Parameters + 
                   NOX::Utils::Details + 
                   NOX::Utils::Warning);
  //
  // Set the nonlinear solver parameters.
  //

  // Line search parameters.
  ParameterList& searchParams = params->sublist ("Line Search");
  searchParams.set ("Method", "More'-Thuente");

  // Parameters for picking the search direction.
  Teuchos::ParameterList& dirParams = params->sublist ("Direction");
  // Use Newton's method to pick the search direction.
  dirParams.set ("Method", "Newton");

  // Parameters for Newton's method.
  ParameterList& newtonParams = dirParams.sublist ("Newton");
  newtonParams.set ("Forcing Term Method", "Constant");
  
  //
  // Newton's method invokes a linear solver repeatedly.
  // Set the parameters for the linear solver.
  //
  ParameterList& lsParams = newtonParams.sublist ("Linear Solver");

  // Use Aztec's implementation of GMRES, with at most 800 iterations,
  // a residual tolerance of 1.0e-4, with output every 50 iterations,
  // and Aztec's native ILU preconditioner.
  lsParams.set ("Aztec Solver", "GMRES");  
  lsParams.set ("Max Iterations", 800);  
  lsParams.set ("Tolerance", 1e-4);
  lsParams.set ("Output Frequency", 50);    
  lsParams.set ("Aztec Preconditioner", "ilu"); 

  RCP<Epetra_CrsMatrix> A = Problem->GetMatrix();

  // Our SimpleProblemInterface implements both Required and
  // Jacobian, so we can use the same object for each.
  RCP<NOX::Epetra::Interface::Required> iReq = interface;
  RCP<NOX::Epetra::Interface::Jacobian> iJac = interface;

  RCP<NOX::Epetra::LinearSystemAztecOO> linSys = 
    rcp (new NOX::Epetra::LinearSystemAztecOO (printParams, lsParams,
                                               iReq, iJac, A, InitialGuess));

  // Need a NOX::Epetra::Vector for constructor.
  NOX::Epetra::Vector noxInitGuess (InitialGuess, NOX::DeepCopy);
  RCP<NOX::Epetra::Group> group = 
    rcp (new NOX::Epetra::Group (printParams, iReq, noxInitGuess, linSys));

  //
  // Set up NOX's iteration stopping criteria ("status tests").
  //

  // ||F(X)||_2 / N < 1.0e-4, where N is the length of F(X).
  //
  // NormF has many options for setting up absolute vs. relative
  // (scaled by the norm of the initial guess) tolerances, scaling or
  // not scaling by the length of F(X), and choosing a different norm
  // (we use the 2-norm here).
  RCP<NOX::StatusTest::NormF> testNormF = 
    rcp (new NOX::StatusTest::NormF (1.0e-4));

  // At most 20 (nonlinear) iterations.
  RCP<NOX::StatusTest::MaxIters> testMaxIters = 
    rcp (new NOX::StatusTest::MaxIters (20));

  // Combine the above two stopping criteria (normwise convergence,
  // and maximum number of nonlinear iterations).  The result tells
  // NOX to stop if at least one of them is satisfied.
  RCP<NOX::StatusTest::Combo> combo = 
    rcp (new NOX::StatusTest::Combo (NOX::StatusTest::Combo::OR, 
                                     testNormF, testMaxIters));

  // Create the NOX nonlinear solver.
  RCP<NOX::Solver::Generic> solver = 
    NOX::Solver::buildSolver (group, combo, params);

  // Solve the nonlinear system.
  NOX::StatusTest::StatusType status = solver->solve();

  // Print the result.
  if (Comm.MyPID() == 0) {
    std::cout << std::endl << "-- Parameter List From Solver --" << std::endl;
    solver->getList ().print (std::cout);
  }

  // Get the Epetra_Vector with the final solution from the solver.
  const NOX::Epetra::Group& finalGroup = 
    dynamic_cast<const NOX::Epetra::Group&> (solver->getSolutionGroup ());

  const Epetra_Vector& finalSolution = 
    dynamic_cast<const NOX::Epetra::Vector&> (finalGroup.getX
    ()).getEpetraVector ();

  // Add a barrier and flush cout, just to make it less likely that
  // the output will get mixed up when running with multiple MPI
  // processes.
  Comm.Barrier ();
  std::cout.flush ();

  if (Comm.MyPID() == 0) {
    std::cout << "Computed solution : " << std::endl;
  }

  // Add a barrier and flush cout, so that the above header line
  // appears before the rest of the vector data.
  Comm.Barrier ();
  std::cout.flush ();

  // Epetra objects know how to print themselves politely when
  // their operator<<(std::ostream&) is invoked on all MPI
  // process(es) in the communicator to which they are associated.
  std::cout << finalSolution;

#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  return EXIT_SUCCESS;
}


