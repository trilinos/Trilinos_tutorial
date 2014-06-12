//
// Simple example of solving the following nonlinear system of
// equations
//
// x(0)^2 + x(1)^2 -1 = 0 
//      x(1) - x(0)^2 = 0
//
// using NOX (Trilinos' Nonlinear Object-Oriented Solutions package).
// For more details and documentation, see the NOX web site:
//
// http://trilinos.sandia.gov/packages/nox/
//
// NOTE: Due to the very small dimension of the problem, it should be
// run with only one MPI process.  We enforce this below by creating a
// subcommunicator containing only MPI Proc 0, and running the problem
// on that communicator, quieting all the others.
//
#include <iostream>

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

#include "NOX.H"
#include "NOX_Epetra_Interface_Required.H"
#include "NOX_Epetra_Interface_Jacobian.H"
#include "NOX_Epetra_LinearSystem_AztecOO.H"
#include "NOX_Epetra_Group.H"

#include "../../aprepro_vhelp.h"

// ==========================================================================
// SimpleProblemInterface, the problem interface in this example,
// defines the interface between NOX and our nonlinear problem to
// solve.
// ==========================================================================
class SimpleProblemInterface : 
  public NOX::Epetra::Interface::Required,
  public NOX::Epetra::Interface::Jacobian
{
public:

  // The constructor accepts an initial guess and the exact solution
  // vector (which we know because we created the example).  We make
  // deep copies of each.
  SimpleProblemInterface (Epetra_Vector& InitialGuess, 
                          Epetra_Vector& ExactSolution) :
    InitialGuess_ (new Epetra_Vector (InitialGuess)),
    ExactSolution_ (new Epetra_Vector (ExactSolution))
  {}

  // Destructor.
  ~SimpleProblemInterface() {}

  // Compute f := F(x), where x is the input vector and f the output
  // vector.
  bool 
  computeF (const Epetra_Vector & x, 
            Epetra_Vector & f,
            NOX::Epetra::Interface::Required::FillType F)
  {
    f[0] = x[0]*x[0] + x[1]*x[1] - 1.0;
    f[1] = x[1] - x[0]*x[0];

    return true;
  };

  bool 
  computeJacobian(const Epetra_Vector & x, Epetra_Operator & Jac)
  {
    Epetra_CrsMatrix* J = dynamic_cast<Epetra_CrsMatrix*>(&Jac);

    if (J == NULL) {
      std::ostringstream os;
      os << "*** Problem_Interface::computeJacobian() - The supplied "
         << "Epetra_Operator object is NOT an Epetra_CrsMatrix! ***";
      throw std::runtime_error (os.str());
    }

    std::vector<int> indices(2);
    std::vector<double> values(2);

    indices[0] = 0; 
    indices[1] = 1;

    // Row 0
    values[0] = 2.0 * x[0];
    values[1] = 2.0 * x[1];
    J->ReplaceGlobalValues (0, 2, &values[0], &indices[0]);

    // Row 1
    values[0] = - 2.0 * x[0];
    values[1] = 1.0;
    J->ReplaceGlobalValues (1, 2, &values[0], &indices[0]);

    return true;
  }

  bool 
  computePrecMatrix (const Epetra_Vector & x, 
                     Epetra_RowMatrix & M) 
  {
    throw std::runtime_error ("*** SimpleProblemInterface does not implement "
                              "computing an explicit preconditioner from an "
                              "Epetra_RowMatrix ***");
  }  

  bool 
  computePreconditioner (const Epetra_Vector & x, 
                         Epetra_Operator & O)
  {
    throw std::runtime_error ("*** SimpleProblemInterface does not implement "
                              "computing an explicit preconditioner from an "
                              "Epetra_Operator ***");
  }  

private:
  Teuchos::RCP<Epetra_Vector> InitialGuess_;
  Teuchos::RCP<Epetra_Vector> ExactSolution_;
};

// =========== //
// main driver //
// =========== //

int 
main (int argc, char **argv)
{
  using Teuchos::ParameterList;
  using Teuchos::parameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
  Epetra_MpiComm CommWorld (MPI_COMM_WORLD);
#else
  Epetra_SerialComm CommWorld;
#endif

  // The example problem is so small that we must run it on only one
  // process.  However, people might run this example code in MPI with
  // any number of processes.  We handle this by using a communicator
  // containing only one MPI process, and quieting all processes but
  // Proc 0 (with respect to MPI_COMM_WORLD).
  if (CommWorld.MyPID() == 0)
    {
#ifdef HAVE_MPI
      Epetra_MpiComm Comm (MPI_COMM_SELF);
#else
      Epetra_SerialComm Comm;
#endif

      // Linear map for the 2 global elements.
      Epetra_Map Map (2, 0, Comm);

      // Build up initial guess and exact solution vectors.
      Epetra_Vector ExactSolution (Map);
      ExactSolution[0] = sqrt (0.5 * (sqrt (5.0) - 1));
      ExactSolution[1] = 0.5 * (sqrt (5.0) - 1);

      Epetra_Vector InitialGuess (Map);
      InitialGuess[0] = 0.5;
      InitialGuess[1] = 0.5;

      // Set up the problem interface.  Your application will define
      // its own problem interface.  SimpleProblemInterface is our
      // example interface, which you can use as a model.
      // 
      // Our SimpleProblemInterface makes a deep copy of the initial
      // guess and exact solution vectors.
      RCP<SimpleProblemInterface> interface = 
        rcp (new SimpleProblemInterface (InitialGuess, ExactSolution));

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

      // Set verbose=true to see a whole lot of intermediate status
      // output, during both linear and nonlinear iterations.
      const bool verbose = false;
      if (verbose) {
        printParams.set ("Output Information", 
                         NOX::Utils::OuterIteration + 
                         NOX::Utils::OuterIterationStatusTest + 
                         NOX::Utils::InnerIteration +
                         NOX::Utils::Parameters + 
                         NOX::Utils::Details + 
                         NOX::Utils::Warning);
      } else {
        printParams.set ("Output Information", NOX::Utils::Warning);
      }

      //
      // Set the nonlinear solver parameters.
      //

      // Line search parameters.
      ParameterList& searchParams = params->sublist ("Line Search");
      searchParams.set ("Method", "Full Step");

      // Parameters for picking the search direction.
      ParameterList& dirParams = params->sublist ("Direction");
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

      // Use Aztec's implementation of GMRES, with at most 800
      // iterations, a residual tolerance of 1.0e-4, with output every
      // 50 iterations, and Aztec's native ILU preconditioner.
      lsParams.set ("Aztec Solver", "GMRES");  
      lsParams.set ("Max Iterations", 800);  
      lsParams.set ("Tolerance", 1e-4);
      lsParams.set ("Output Frequency", 50);    
      lsParams.set ("Aztec Preconditioner", "ilu"); 

      //
      // Build the Jacobian matrix.
      //
      RCP<Epetra_CrsMatrix> A = rcp (new Epetra_CrsMatrix (Copy, Map, 2));
      {
        std::vector<int> indices(2);
        std::vector<double> values(2);

        indices[0]=0; 
        indices[1]=1;

        values[0] = 2.0 * InitialGuess[0];
        values[1] = 2.0 * InitialGuess[1];
        A.get()->InsertGlobalValues (0, 2, &values[0], &indices[0]);

        values[0] = -2.0 * InitialGuess[0];
        values[1] = 1.0;
        A.get()->InsertGlobalValues (1, 2, &values[0], &indices[0]);

        A.get()->FillComplete();
      }  

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
      // (scaled by the norm of the initial guess) tolerances, scaling
      // or not scaling by the length of F(X), and choosing a
      // different norm (we use the 2-norm here).
      RCP<NOX::StatusTest::NormF> testNormF = 
        rcp (new NOX::StatusTest::NormF (1.0e-4));

      // At most 20 (nonlinear) iterations.
      RCP<NOX::StatusTest::MaxIters> testMaxIters = 
        rcp (new NOX::StatusTest::MaxIters (20));

      // Combine the above two stopping criteria (normwise
      // convergence, and maximum number of nonlinear iterations).
      // The result tells NOX to stop if at least one of them is
      // satisfied.
      RCP<NOX::StatusTest::Combo> combo = 
        rcp (new NOX::StatusTest::Combo (NOX::StatusTest::Combo::OR, 
                                         testNormF, testMaxIters));

      // Create the NOX nonlinear solver.
      RCP<NOX::Solver::Generic> solver = 
        NOX::Solver::buildSolver (group, combo, params);

      // Solve the nonlinear system.
      NOX::StatusTest::StatusType status = solver->solve();

      // Print the result.
      //
      // For this particular example, Comm contains only one MPI
      // process.  However, we check for Comm.MyPID() == 0 here just
      // so that the example is fully general.  (If you're solving a
      // larger nonlinear problem, you could safely use the code
      // below.)
      if (Comm.MyPID() == 0) {
        std::cout << std::endl << "-- Parameter List From Solver --" << std::endl;
        solver->getList ().print (std::cout);
      }

      // Get the Epetra_Vector with the final solution from the solver.
      const NOX::Epetra::Group& finalGroup = 
        dynamic_cast<const NOX::Epetra::Group&>(solver->getSolutionGroup());

      const Epetra_Vector& finalSolution = 
        dynamic_cast<const NOX::Epetra::Vector&> (finalGroup.getX ()).getEpetraVector ();

      if (Comm.MyPID() == 0) {
        std::cout << "Computed solution: " << std::endl;
      }
      // Epetra objects know how to print themselves politely when
      // their operator<<(std::ostream&) is invoked on all MPI
      // process(es) in the communicator to which they are associated.
      std::cout << finalSolution;

      if (Comm.MyPID() == 0) {
        std::cout << "Exact solution: " << std::endl;
      }
      std::cout << ExactSolution;
    }

  // Remember how we quieted all MPI processes but Proc 0 above?
  // Now we're back in MPI_COMM_WORLD again.
#ifdef HAVE_MPI
  // Make sure that everybody is done before calling MPI_Finalize().
  MPI_Barrier (MPI_COMM_WORLD);
  MPI_Finalize();
#endif
  return EXIT_SUCCESS;
}


