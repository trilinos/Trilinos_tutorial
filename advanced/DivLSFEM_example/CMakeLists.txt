include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#Add Trilinos information to the include and link lines
include_directories(${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS} )
link_directories(${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS} )

#set trilinos libraries to link (LINK_LIBRARIES)
set(LINK_LIBRARIES ${ML_LIBRARIES} ${Aztecoo_LIBRARIES} ${Pamgen_LIBRARIES} ${Epetraext_LIBRARIES} ${Shards_LIBRARIES} ${Teuchos_LIBRARIES} ${Epetra_LIBRARIES} ${Intrepid_LIBRARIES})

add_executable(DivLSFEM_example example_DivLSFEM.cpp)
target_link_libraries(DivLSFEM_example ${LINK_LIBRARIES})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/DivLSFEMin.xml ${CMAKE_CURRENT_BINARY_DIR}/DivLSFEMin.xml COPYONLY)

####
INCLUDE(Dart)
INCLUDE(CPack)

