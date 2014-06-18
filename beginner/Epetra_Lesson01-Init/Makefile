# Get Trilinos as one entity
include $(TRILINOS)/include/Makefile.export.Trilinos
#include $(TRILINOS)/include/Makefile.export.Anasazi

# Make sure to use same compilers and flags as Trilinos
CXX=$(Trilinos_CXX_COMPILER)
CC=$(Trilinos_C_COMPILER)
FORT=$(Trilinos_Fortran_COMPILER)

# Correctly set compilation flags to include both trilinos' flags and flags defined by the user
CXX_FLAGS=$(Trilinos_CXX_COMPILER_FLAGS) $(USER_CXX_FLAGS)
C_FLAGS=$(Trilinos_C_COMPILER_FLAGS) $(USERC_FLAGS)
FORT_FLAGS=$(Trilinos_Fortran_COMPILER_FLAGS) $(USER_FORT_FLAGS)

# set of libraries to correctly link to the target
INCLUDE_DIRS=$(Trilinos_INCLUDE_DIRS) $(Trilinos_TPL_INCLUDE_DIRS)
LIBRARY_DIRS=$(Trilinos_LIBRARY_DIRS) $(Trilinos_TPL_LIBRARY_DIRS)
LIBRARIES=$(Trilinos_LIBRARIES) $(Trilinos_TPL_LIBRARIES)


LINK_FLAGS=$(Trilinos_EXTRA_LD_FLAGS)

#just assuming that epetra is turned on.
#DEFINES=-DMYAPP_EPETRA -DHAVE_MPI
DEFINES=-DHAVE_MPI

# targets to be built with the default "make" command
default: print_info Epetra_lesson01_mpi_on_its_own Epetra_lesson01_mpi_only_through_Epetra Epetra_lesson01_no_mpi

# Echo trilinos build info just for fun
print_info:
	@echo " Found Trilinos!  Here are the details: "
	@echo "   Trilinos_VERSION = $(Trilinos_VERSION)"
	@echo "   Trilinos_PACKAGE_LIST = $(Trilinos_PACKAGE_LIST)"
	@echo "   Trilinos_LIBRARIES = $(Trilinos_LIBRARIES)"
	@echo "   Trilinos_INCLUDE_DIRS = $(Trilinos_INCLUDE_DIRS)"
	@echo "   Trilinos_LIBRARY_DIRS = $(Trilinos_LIBRARY_DIRS)"
	@echo "   Trilinos_TPL_LIST = $(Trilinos_TPL_LIST)"
	@echo "   Trilinos_TPL_INCLUDE_DIRS = $(Trilinos_TPL_INCLUDE_DIRS)"
	@echo "   Trilinos_TPL_LIBRARIES = $(Trilinos_TPL_LIBRARIES)"
	@echo "   Trilinos_TPL_LIBRARY_DIRS = $(Trilinos_TPL_LIBRARY_DIRS)"
	@echo "   Trilinos_BUILD_SHARED_LIBS = $(Trilinos_BUILD_SHARED_LIBS)"
	@echo "End of Trilinos details"

# run the given tests
test: Epetra_lesson01_mpi_on_its_own Epetra_lesson01_mpi_only_through_Epetra Epetra_lesson01_no_mpi
	./Epetra_lesson01_mpi_on_its_own
	./Epetra_lesson01_mpi_only_through_Epetra
	./Epetra_lesson01_no_mpi

# build the executables by linking the object code to the necessary libraries
Epetra_lesson01_mpi_on_its_own: lesson01_mpi_on_its_own.o
	$(CXX) $(CXX_FLAGS) lesson01_mpi_on_its_own.o -o Epetra_lesson01_mpi_on_its_own $(LINK_FLAGS) $(INCLUDE_DIRS) $(DEFINES) $(LIBRARY_DIRS) $(LIBRARIES)

Epetra_lesson01_mpi_only_through_Epetra: lesson01_mpi_only_through_Epetra.o
	$(CXX) $(CXX_FLAGS) lesson01_mpi_only_through_Epetra.o -o Epetra_lesson01_mpi_only_through_Epetra $(LINK_FLAGS) $(INCLUDE_DIRS) $(DEFINES) $(LIBRARY_DIRS) $(LIBRARIES)

Epetra_lesson01_no_mpi: lesson01_no_mpi.o
	$(CXX) $(CXX_FLAGS) lesson01_no_mpi.o -o Epetra_lesson01_no_mpi $(LINK_FLAGS) $(INCLUDE_DIRS) $(DEFINES) $(LIBRARY_DIRS) $(LIBRARIES)

# compile source code into the object code using the pre-defined flags
lesson01_mpi_on_its_own.o:
	$(CXX) -c $(CXX_FLAGS) $(INCLUDE_DIRS) $(DEFINES) lesson01_mpi_on_its_own.cpp


lesson01_mpi_only_through_Epetra.o:
	$(CXX) -c $(CXX_FLAGS) $(INCLUDE_DIRS) $(DEFINES) lesson01_mpi_only_through_Epetra.cpp

lesson01_no_mpi.o:
	$(CXX) -c $(CXX_FLAGS) $(INCLUDE_DIRS) $(DEFINES) lesson01_no_mpi.cpp
.PHONY: clean
clean:
	rm -f *.o *.a Epetra_lesson01_mpi_on_its_own Epetra_lesson01_mpi_only_through_Epetra Epetra_lesson01_no_mpi
