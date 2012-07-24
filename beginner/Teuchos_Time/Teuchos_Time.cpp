// You can include this header file whether or not you built with MPI.
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_oblackholestream.hpp"

#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_Version.hpp"

#include "../../aprepro_vhelp.h"

//
// Evaluate a quadratic function at x.
//
double quadFunc (int x);

//
// Compute the factorial of x.
//
double factFunc (int x);

//
// Global timers.  These will be instantiated in main(), before
// calling quadFunc() or factFunc().  They are declared here 
// because quadFunc() resp. factFunc() refer to them.
//
Teuchos::RCP<Teuchos::Time> CompTime;
Teuchos::RCP<Teuchos::Time> FactTime;

//
// The main() driver routine.
//
int 
main (int argc, char* argv[])
{
  using std::endl;
  using Teuchos::RCP;
  using Teuchos::Time;
  using Teuchos::TimeMonitor;

  // Teuchos::GlobalMPISession's destructor automatically calls 
  // MPI_Init() if appropriate.  Passing in NULL as the third
  // argument is a good idea when running with a large number 
  // of MPI processes; it silences the default behavior of 
  // printing out a short message on each MPI process.
  Teuchos::GlobalMPISession mpiSession (&argc, &argv, NULL);
  const int procRank = Teuchos::GlobalMPISession::getRank();

  // Only let MPI Proc 0 print to stdout.  The other processes
  // processes print to a "black hole," which is like /dev/null;
  // it doesn't display any output.
  Teuchos::oblackholestream blackhole;
  std::ostream &out = (procRank == 0 ? std::cout : blackhole);

  out << Teuchos::Teuchos_Version() << endl << endl;

  // Create the global timers.
  CompTime = TimeMonitor::getNewCounter ("Computational Time");
  FactTime = TimeMonitor::getNewCounter ("Factorial Time");

  // Apply the quadratic function.  We'll time this with CompTime.
  {
    double x;
    for (int i = -100; i < 100; ++i) {
      x = quadFunc (i);
    }
  }

  // Apply the factorial function.  We'll time this with FactTime.
  {
    double x;
    for (int i = 0; i < 100; ++i) {
      x = factFunc (i);
    }
  }

  // Get a summary of timings over all MPI processes.
  // Only MPI Proc 0 gets to print anything anyway, so passing in
  // "out" is OK.
  TimeMonitor::summarize (out);

  // Teuchos::GlobalMPISession's destructor automatically calls 
  // MPI_Finalize() if appropriate.
  return 0;
}

double 
quadFunc (int x)
{
  // Construct a local time monitor.  
  // This starts the CompTime timer and will stop when the scope exits.
  Teuchos::TimeMonitor LocalTimer (*CompTime);

  // Evaluate the quadratic function.
  return x*x - 1.0;
}

double 
factFunc (int x)
{
  // Construct a local time monitor.
  // This starts the FactTime timer and will stop when the scope exits.
  Teuchos::TimeMonitor LocalTimer (*FactTime);

  if (x == 0) 
    return 0.0;
  else if (x == 1)
    return 1.0;
  else // Compute the factorial recursively.
    return x * factFunc (x-1);
}
