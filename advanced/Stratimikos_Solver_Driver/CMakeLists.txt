include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#Add Trilinos information to the include and link lines
include_directories(${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS} )
link_directories(${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS} )

#set trilinos libraries to link (LINK_LIBRARIES)
set(LINK_LIBRARIES ${Epetra_LIBRARIES} ${Galeri_LIBRARIES} ${Ifpack_LIBRARIES} ${Belos_LIBRARIES} ${Tpetra_LIBRARIES} ${Stratimikos_LIBRARIES} ${Thyra_LIBRARIES} ${Teuchos_LIBRARIES})


add_executable(Stratimikos_Solver_Driver
    test_single_stratimikos_solver_driver.cpp
#    test_single_stratimikos_solver.hpp
    test_single_stratimikos_solver.cpp)
target_link_libraries(Stratimikos_Solver_Driver ${LINK_LIBRARIES})


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/FourByFour.mtx ${CMAKE_CURRENT_BINARY_DIR}/FourByFour.mtx COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/FourByFour.xml ${CMAKE_CURRENT_BINARY_DIR}/FourByFour.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/FourByFour.amesos.xml ${CMAKE_CURRENT_BINARY_DIR}/FourByFour.amesos.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/FourByFour.aztecoo.np.xml ${CMAKE_CURRENT_BINARY_DIR}/FourByFour.aztecoo.np.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/FourByFour.aztecoo.ifpack.xml ${CMAKE_CURRENT_BINARY_DIR}/FourByFour.aztecoo.ifpack.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/FourByFour.aztecoo.ifpack.adjoint.xml ${CMAKE_CURRENT_BINARY_DIR}/FourByFour.aztecoo.ifpack.adjoint.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/FourByFour.aztecoo.ml.xml ${CMAKE_CURRENT_BINARY_DIR}/FourByFour.aztecoo.ml.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/FourByFour.belos.np.xml ${CMAKE_CURRENT_BINARY_DIR}/FourByFour.belos.np.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/FourByFour.belos.ifpack.xml ${CMAKE_CURRENT_BINARY_DIR}/FourByFour.belos.ifpack.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/FourByFour.belos.ml.xml ${CMAKE_CURRENT_BINARY_DIR}/FourByFour.belos.ml.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/BelosGCRODRTest.xml ${CMAKE_CURRENT_BINARY_DIR}/BelosGCRODRTest.xml COPYONLY)


####
INCLUDE(Dart)
INCLUDE(CPack)

