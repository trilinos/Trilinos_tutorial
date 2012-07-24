//@HEADER
// ************************************************************************
// 
//               Tpetra: Linear Algebra Services Package 
//                 Copyright (2009) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//  
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//  
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ************************************************************************
//@HEADER

//
// Example of basic initialization boilerplate for using Tpetra.
//
// Includes MPI initialization, getting a Teuchos::Comm communicator,
// and printing out Tpetra version information.
//

#include <Tpetra_DefaultPlatform.hpp>
#include <Tpetra_Version.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_oblackholestream.hpp>

#include "../../aprepro_vhelp.h"

void
exampleRoutine (const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
                std::ostream& out)
{
  using std::endl;

  // Print out the Tpetra software version information.
  out << Tpetra::version() << endl << endl;
}


int 
main (int argc, char *argv[]) 
{
  // These "using" declarations make the code more concise, in that
  // you don't have to write the namespace along with the class or
  // object name.  This is especially helpful with commonly used
  // things like std::endl or Teuchos::RCP.
  using std::endl;
  using Teuchos::RCP;

  // A "black hole stream" prints nothing.  It's like /dev/null in
  // Unix-speak.  The typical MPI convention is that only MPI Rank 0
  // is allowed to print anything.  We enforce this convention by
  // setting Rank 0 to use std::cout for output, but all other ranks
  // to use the black hole stream.  It's more concise and less error
  // prone than having to check the rank every time you want to print.
  Teuchos::oblackholestream blackHole;

  // Start up MPI, if using MPI.  Trilinos doesn't have to be built
  // with MPI; it's called a "serial" build if you build without MPI.
  // GlobalMPISession hides this implementation detail.
  //
  // Note the third argument.  If you pass GlobalMPISession the
  // address of an std::ostream, it will print a one-line status
  // message with the rank on each MPI process.  This may be
  // undesirable if running with a large number of MPI processes.  You
  // can avoid printing anything here by passing in either a "black
  // hole stream" (see above) or NULL.
  Teuchos::GlobalMPISession mpiSession (&argc, &argv, &blackHole);

  // Get a pointer to the communicator object representing
  // MPI_COMM_WORLD.  getDefaultPlatform.getComm() doesn't create a
  // new object every time you call it; it just returns the same
  // communicator each time.  Thus, you can call it anywhere and get
  // the same communicator.  (This is handy if you don't want to pass
  // a communicator around everywhere, though it's always better to
  // parameterize your algorithms on the communicator.)
  //
  // "Tpetra::DefaultPlatform" knows whether or not we built with MPI
  // support.  If we didn't build with MPI, we'll get a "communicator"
  // with size 1, whose only process has rank 0.
  RCP<const Teuchos::Comm<int> > comm = 
    Tpetra::DefaultPlatform::getDefaultPlatform().getComm();

  const int myRank = comm->getRank();
  const int numProcs = comm->getSize();

  // The stream to which to write output.  Only MPI Rank 0 gets to
  // write to stdout; the other MPI processes get a "black hole
  // stream" (see above).
  std::ostream& out = (myRank == 0) ? std::cout : blackHole;

  // We have a communicator and an output stream.
  // Let's do something with them!
  exampleRoutine (comm, out);

  // GlobalMPISession calls MPI_Finalize() in its destructor, if
  // appropriate.  You don't have to do anything here!  Just return
  // from main().  Isn't that helpful?
  return 0;
}
