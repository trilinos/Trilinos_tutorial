include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#Add Trilinos information to the include and link lines
include_directories(${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS} )
link_directories(${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS} )
# /Library/Frameworks/QtCore.framework /Library/Frameworks/QtGui.framework)

#add executable
add_executable(Epetra_lesson04_sparse_matrix_fill lesson04_sparse_matrix_fill.cpp)
target_link_libraries(Epetra_lesson04_sparse_matrix_fill ${Epetra_LIBRARIES})
add_test(Epetra_lesson04_sparse_matrix_fill ${EXECUTABLE_OUTPUT_PATH}/Epetra_lesson04_sparse_matrix_fill)

INCLUDE(Dart)
INCLUDE(CPack)

