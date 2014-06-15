include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#Add Trilinos information to the include and link lines
include_directories(${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS} )
link_directories(${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS} )
# /Library/Frameworks/QtCore.framework /Library/Frameworks/QtGui.framework)
#set trilinos libraries to link (LINK_LIBRARIES)
set(LINK_LIBRARIES ${Tpetra_LIBRARIES} ${Teuchos_LIBRARIES})

#add executable
add_executable(Tpetra_lesson01_mpi_only_through_Tpetra lesson01_mpi_only_through_Tpetra.cpp)
target_link_libraries(Tpetra_lesson01_mpi_only_through_Tpetra ${LINK_LIBRARIES})
add_test(Tpetra_lesson01_mpi_only_through_Tpetra ${EXECUTABLE_OUTPUT_PATH}/Tpetra_lesson01_mpi_only_through_Tpetra)

add_executable(Tpetra_lesson01_mpi_on_its_own lesson01_mpi_on_its_own.cpp)
target_link_libraries(Tpetra_lesson01_mpi_on_its_own ${LINK_LIBRARIES})
add_test(Tpetra_lesson01_mpi_on_its_own ${EXECUTABLE_OUTPUT_PATH}/Tpetra_lesson01_mpi_on_its_own)

add_executable(Tpetra_lesson01_no_mpi lesson01_no_mpi.cpp)
target_link_libraries(Tpetra_lesson01_no_mpi ${LINK_LIBRARIES})
add_test(Tpetra_lesson01_no_mpi ${EXECUTABLE_OUTPUT_PATH}/Tpetra_lesson01_no_mpi)

INCLUDE(Dart)
INCLUDE(CPack)

