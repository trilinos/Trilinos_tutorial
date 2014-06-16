include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#Add Trilinos information to the include and link lines
include_directories(${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS} )
link_directories(${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS} )
# /Library/Frameworks/QtCore.framework /Library/Frameworks/QtGui.framework)

#add executable
add_executable(Epetra_lesson01_mpi_only_through_Epetra lesson01_mpi_only_through_Epetra.cpp)
target_link_libraries(Epetra_lesson01_mpi_only_through_Epetra ${Epetra_LIBRARIES})
add_test(Epetra_lesson01_mpi_only_through_Epetra ${EXECUTABLE_OUTPUT_PATH}/Epetra_lesson01_mpi_only_through_Epetra)

add_executable(Epetra_lesson01_mpi_on_its_own lesson01_mpi_on_its_own.cpp)
target_link_libraries(Epetra_lesson01_mpi_on_its_own ${Epetra_LIBRARIES})
add_test(Epetra_lesson01_mpi_on_its_own ${EXECUTABLE_OUTPUT_PATH}/Epetra_lesson01_mpi_on_its_own)

add_executable(Epetra_lesson01_no_mpi lesson01_no_mpi.cpp)
target_link_libraries(Epetra_lesson01_no_mpi ${Epetra_LIBRARIES})
add_test(Epetra_lesson01_no_mpi ${EXECUTABLE_OUTPUT_PATH}/Epetra_lesson01_no_mpi)

INCLUDE(Dart)
INCLUDE(CPack)

