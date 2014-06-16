include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#Add Trilinos information to the include and link lines
include_directories(${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS} )
link_directories(${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS} )
# /Library/Frameworks/QtCore.framework /Library/Frameworks/QtGui.framework)

#add executable
add_executable(Epetra_lesson02_read_modify_vec lesson02_read_modify_vec.cpp)
target_link_libraries(Epetra_lesson02_read_modify_vec ${Epetra_LIBRARIES})
add_test(Epetra_lesson02_read_modify_vec ${EXECUTABLE_OUTPUT_PATH}/Epetra_lesson02_read_modify_vec)

add_executable(Epetra_lesson02_init_map_vec lesson02_init_map_vec.cpp)
target_link_libraries(Epetra_lesson02_init_map_vec ${Epetra_LIBRARIES})
add_test(Epetra_lesson02_init_map_vec ${EXECUTABLE_OUTPUT_PATH}/Epetra_lesson02_init_map_vec)

INCLUDE(Dart)
INCLUDE(CPack)

