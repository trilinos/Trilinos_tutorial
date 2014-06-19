include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#Add Trilinos information to the include and link lines
include_directories(${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS} )
link_directories(${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS} )
# /Library/Frameworks/QtCore.framework /Library/Frameworks/QtGui.framework)

#set trilinos libraries to link (LINK_LIBRARIES)
set(LINK_LIBRARIES ${Epetra_LIBRARIES} ${Galeri_LIBRARIES} ${Teuchos_LIBRARIES} ${Ifpack_LIBRARIES} ${Belos_LIBRARIES} ${Tpetra_LIBRARIES})
#./epetra_test_err.h)

#add executable + test
add_executable(Epetra_Basic_Perf_Test Epetra_Basic_Perf_Test.cpp)
target_link_libraries(Epetra_Basic_Perf_Test  ${LINK_LIBRARIES})
add_test(Epetra_Basic_Perf_test Epetra_Basic_Perf_Test 16 12 1 1 25 -v)
add_test(Epetra_Basic_Perf_mpi mpiexec -np 1 Epetra_Basic_Perf_Test 16 12 1 1 25 -v)
add_test(Epetra_Basic_Perf_mpi_2Procs mpiexec -np 2 Epetra_Basic_Perf_Test 16 12 2 1 25 -v)
add_test(Epetra_Basic_Perf_mpi_moreProcs mpiexec -np 15 Epetra_Basic_Perf_Test 20 30 5 3 25 -v)

add_executable(Epetra_CrsMatrix Epetra_CrsMatrix.cpp)
target_link_libraries(Epetra_CrsMatrix  ${LINK_LIBRARIES})
add_test(Epetra_CrsMatrix_test Epetra_CrsMatrix -v)
add_test(Epetra_CrsMatrix_test_mpi_1proc mpiexec Epetra_CrsMatrix -v)
#add_test(Epetra_CrsMatrix_test_mpi_3proc mpiexec -np 3 Epetra_CrsMatrix -v)
add_test(Epetra_CrsMatrix_test_mpi_7proc mpiexec -np 7 Epetra_CrsMatrix -v)
#add_test(Epetra_CrsMatrix_test_mpi_15proc mpiexec -np 15 Epetra_CrsMatrix -v)
#add_test(Epetra_CrsMatrix_test_mpi_20proc mpiexec -np 20 Epetra_CrsMatrix -v)


#ADD_SUBDIRECTORY(Belos_Block)
ADD_SUBDIRECTORY(Epetraext_MatrixMatrix)
ADD_SUBDIRECTORY(Stratimikos_Preconditioner)
ADD_SUBDIRECTORY(Stratimikos_Solver_Driver)
ADD_SUBDIRECTORY(Teuchos_UnitTest)
ADD_SUBDIRECTORY(Teuchos_ParameterList)
#ADD_SUBDIRECTORY(Teuchos_ParameterList-10_8)
#ADD_SUBDIRECTORY(Tpetra_Basic_PerfTest)

INCLUDE(Dart)
INCLUDE(CPack)

