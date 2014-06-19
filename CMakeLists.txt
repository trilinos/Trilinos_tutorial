cmake_minimum_required(VERSION 2.6)

MESSAGE("Starting configure")

# Use TRILINOS_PATH if the user set it, to help find Trilinos 10.7 or greater.
# The final location will actually be held in Trilinos_DIR which must
# point at "<prefix>/lib/cmake/Trilinos", but this helps the search.
SET(CMAKE_PREFIX_PATH ${TRILINOS_PATH} ${CMAKE_PREFIX_PATH})

#Look for Trilinos - the first PATH will find Trilinos 10.4 and 10.6,
#the second PATH will find Trilinos 10.2
FIND_PACKAGE(Trilinos REQUIRED PATHS ${TRILINOS_PATH}/include ${TRILINOS_PATH})

#Output some info about the Trilinos installation found
IF(Trilinos_FOUND)
   MESSAGE("\nFound Trilinos!  Here are the details: ")
   MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
   MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
   MESSAGE("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
   MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
   MESSAGE("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
   MESSAGE("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
   MESSAGE("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
   MESSAGE("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
   # Will Tramonto work with a Shared version of Trilinos?
   MESSAGE("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
   #These last outputs are temporary
   MESSAGE("   Trilinos_CXX_COMPILER = ${Trilinos_CXX_COMPILER}")
   MESSAGE("   Trilinos_C_COMPILER = ${Trilinos_C_COMPILER}")
   MESSAGE("   Trilinos_Fortran_COMPILER = ${Trilinos_Fortran_COMPILER}")
   MESSAGE("   Trilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
   MESSAGE("   Trilinos_C_COMPILER_FLAGS = ${Trilinos_C_COMPILER_FLAGS}")
   MESSAGE("   Trilinos_Fortran_COMPILER_FLAGS =
     ${Trilinos_Fortran_COMPILER_FLAGS}")
   MESSAGE("   Trilinos_LINKER = ${Trilinos_LINKER}")
   MESSAGE("   Trilinos_EXTRA_LD_FLAGS = ${Trilinos_EXTRA_LD_FLAGS}")
   MESSAGE("   Trilinos_AR = ${Trilinos_AR}")
   # Add MPI_EXEC, MPI_EXEC_MAX_NUMPROCS, MPI_EXEC_NUMPROC_FLAGS
   # These are not available in Trilinos 10.2.0, but should be in 10.4.0
   MESSAGE("End of Trilinos details\n")
ELSE()
  # We should never reach this because FIND_PACKAGE(Trilinos now uses REQUIRED
  MESSAGE(FATAL_ERROR "Could not find Trilinos 10.2 or greater!")
ENDIF()

IF(NOT CMAKE_CXX_COMPILER)
  MESSAGE("Setting CXX compiler.")
  SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
ENDIF()

IF(NOT CMAKE_C_COMPILER)
  MESSAGE("Setting C compiler.")
  SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER})
ENDIF()

IF(NOT CMAKE_LINKER)
  MESSAGE("Setting linker.")
  SET(CMAKE_LINKER ${Trilinos_LINKER})
ENDIF()

IF(NOT CMAKE_EXE_LINKER_FLAGS)
  MESSAGE("Setting linker flags.")
  SET(CMAKE_EXE_LINKER_FLAGS ${Trilinos_EXTRA_LD_FLAGS})
ENDIF()

IF(NOT CMAKE_AR)
  MESSAGE("Setting ar.")
  SET(CMAKE_AR ${CMAKE_AR})
ENDIF()

project(trilinos_tutorial)
SET(CPACK_PACKAGE_VERSION "1.0")
set(TRILINOS_PATH "" CACHE FILEPATH "Path to Trilinos Directory")

IF(NOT CMAKE_CXX_FLAGS)
  MESSAGE("Setting CXX compiler flags.")
  SET(CMAKE_CXX_FLAGS ${Trilinos_CXX_COMPILER_FLAGS})
ENDIF()

IF(NOT CMAKE_C_FLAGS)
  MESSAGE("Setting C compiler flags.")
  SET(CMAKE_C_FLAGS ${Trilinos_C_COMPILER_FLAGS})
ENDIF()

ENABLE_TESTING()

ADD_SUBDIRECTORY(advanced)
ADD_SUBDIRECTORY(beginner)
#ADD_SUBDIRECTORY(custom)
ADD_SUBDIRECTORY(ctests)
ADD_SUBDIRECTORY(gui/tutorial_gui)

include(Dart)
include(CPack)
