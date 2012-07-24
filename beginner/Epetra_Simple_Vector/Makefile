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
default: print_info Epetra_Simple_Vector

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

# run the given test
test: Epetra_Simple_Vector input.xml
	./Epetra_Simple_Vector

# build the executable by linking the object code to the necessary libraries
Epetra_Simple_Vector: Epetra_Simple_Vector.o
	$(CXX) $(CXX_FLAGS) Epetra_Simple_Vector.o -o Epetra_Simple_Vector $(LINK_FLAGS) $(INCLUDE_DIRS) $(DEFINES) $(LIBRARY_DIRS) $(LIBRARIES)

# compile source code into the object code using the pre-defined flags
Epetra_Simple_Vector.o:
	$(CXX) -c $(CXX_FLAGS) $(INCLUDE_DIRS) $(DEFINES) Epetra_Simple_Vector.cpp
.PHONY: clean
clean:
	rm -f *.o *.a Epetra_Simple_Vector
