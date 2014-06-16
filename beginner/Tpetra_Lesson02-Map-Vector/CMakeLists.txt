include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#Add Trilinos information to the include and link lines
include_directories(${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS} )
link_directories(${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS} )
# /Library/Frameworks/QtCore.framework /Library/Frameworks/QtGui.framework)
#set trilinos libraries to link (LINK_LIBRARIES)
set(LINK_LIBRARIES ${Tpetra_LIBRARIES} ${Teuchos_LIBRARIES})

#add executable
add_executable(Tpetra_lesson02_read_modify_vec lesson02_read_modify_vec.cpp)
target_link_libraries(Tpetra_lesson02_read_modify_vec ${LINK_LIBRARIES})
add_test(Tpetra_lesson02_read_modify_vec ${EXECUTABLE_OUTPUT_PATH}/Tpetra_lesson02_read_modify_vec)

add_executable(Tpetra_lesson02_init_map_vec lesson02_init_map_vec.cpp)
target_link_libraries(Tpetra_lesson02_init_map_vec ${LINK_LIBRARIES})
add_test(Tpetra_lesson02_init_map_vec ${EXECUTABLE_OUTPUT_PATH}/Tpetra_lesson02_init_map_vec)

INCLUDE(Dart)
INCLUDE(CPack)

