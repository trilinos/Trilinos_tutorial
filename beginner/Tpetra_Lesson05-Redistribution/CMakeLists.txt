include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#Add Trilinos information to the include and link lines
include_directories(${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS} )
link_directories(${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS} )
# /Library/Frameworks/QtCore.framework /Library/Frameworks/QtGui.framework)
#set trilinos libraries to link (LINK_LIBRARIES)
set(LINK_LIBRARIES ${Tpetra_LIBRARIES} ${Teuchos_LIBRARIES})

#add executable
add_executable(Tpetra_lesson05_redistribution lesson05_redistribution.cpp)
target_link_libraries(Tpetra_lesson05_redistribution ${LINK_LIBRARIES})
add_test(Tpetra_lesson05_redistribution ${EXECUTABLE_OUTPUT_PATH}/Tpetra_lesson05_redistribution)

INCLUDE(Dart)
INCLUDE(CPack)

