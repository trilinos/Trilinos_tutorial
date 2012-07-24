include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#Add Trilinos information to the include and link lines
include_directories(${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS} )
link_directories(${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS} )
# /Library/Frameworks/QtCore.framework /Library/Frameworks/QtGui.framework)

#set trilinos libraries to link (LINK_LIBRARIES)
set(LINK_LIBRARIES ${Epetra_LIBRARIES} ${Galeri_LIBRARIES} ${Teuchos_LIBRARIES} ${Ifpack_LIBRARIES} ${Belos_LIBRARIES})

#add executable
add_executable(Ifpack_Preconditioner_Factory Ifpack_Preconditioner_Factory.cpp)
target_link_libraries(Ifpack_Preconditioner_Factory  ${LINK_LIBRARIES})

INCLUDE(CPack)

