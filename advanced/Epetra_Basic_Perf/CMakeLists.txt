include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#Add Trilinos information to the include and link lines
include_directories(${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS} )
link_directories(${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS} )
# /Library/Frameworks/QtCore.framework /Library/Frameworks/QtGui.framework)

#set trilinos libraries to link (LINK_LIBRARIES)
set(LINK_LIBRARIES ${Epetra_LIBRARIES} ${Galeri_LIBRARIES} ${Teuchos_LIBRARIES} ${Ifpack_LIBRARIES} ${Belos_LIBRARIES} ${Tpetra_LIBRARIES})
#./epetra_test_err.h)

#add executable + test
add_executable(Epetra_Basic_Perf Epetra_Basic_Perf.cpp)
target_link_libraries(Epetra_Basic_Perf  ${LINK_LIBRARIES})

INCLUDE(Dart)
INCLUDE(CPack)

