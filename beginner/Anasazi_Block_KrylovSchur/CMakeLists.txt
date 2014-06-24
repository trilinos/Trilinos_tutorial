include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#Add Trilinos information to the include and link lines
include_directories(${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS} )
link_directories(${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS} )
# /Library/Frameworks/QtCore.framework /Library/Frameworks/QtGui.framework)

#set trilinos libraries to link (LINK_LIBRARIES)
set(LINK_LIBRARIES ${Epetra_LIBRARIES} ${Anasazi_LIBRARIES} ${Amesos_LIBRARIES})

#add executable
add_executable(Anasazi_Block_KrylovSchur Anasazi_Block_KrylovSchur.cpp)
add_executable(Anasazi_Block_KrylovSchur_Amesos Anasazi_Block_KrylovSchur_Amesos.cpp)
target_link_libraries(Anasazi_Block_KrylovSchur ${LINK_LIBRARIES})
target_link_libraries(Anasazi_Block_KrylovSchur_Amesos ${LINK_LIBRARIES})

INCLUDE(CPack)

