include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#Add Trilinos information to the include and link lines
include_directories(${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS} )
link_directories(${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS} )

#set trilinos libraries to link (LINK_LIBRARIES)
set(LINK_LIBRARIES ${Epetra_LIBRARIES} ${Galeri_LIBRARIES} ${Ifpack_LIBRARIES} ${Belos_LIBRARIES} ${Tpetra_LIBRARIES} ${Stratimikos_LIBRARIES} ${Thyra_LIBRARIES} ${Teuchos_LIBRARIES})

####


#add_executable(Belos_GCRODR_strattest test_belos_gcrodr.cpp)
#target_link_libraries(Belos_GCRODR_strattest ${LINK_LIBRARIES} )
#add_test(Belos_GCRODR_strattest Belos_GCRODR_strattest --show-timer-summary)
#add_test(Belos_GCRODR_strattest_mpi mpiexec -np 4 Belos_GCRODR_strattest --show-timer-summary)

add_executable(test_single_stratimikos_solver_driver 
    test_single_stratimikos_solver_driver.cpp
#    test_single_stratimikos_solver.hpp
    test_single_stratimikos_solver.cpp)
target_link_libraries(test_single_stratimikos_solver_driver ${LINK_LIBRARIES})


#/#

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/In_FourByFour.mtx ${CMAKE_CURRENT_BINARY_DIR}/FourByFour.mtx COPYONLY)

#copy data file over
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/_FourByFour.xml ${CMAKE_CURRENT_BINARY_DIR}/FourByFour.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/_FourByFour.amesos.xml ${CMAKE_CURRENT_BINARY_DIR}/FourByFour.amesos.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/_FourByFour.aztecoo.np.xml ${CMAKE_CURRENT_BINARY_DIR}/FourByFour.aztecoo.np.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/_FourByFour.aztecoo.ifpack.xml ${CMAKE_CURRENT_BINARY_DIR}/FourByFour.aztecoo.ifpack.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/_FourByFour.aztecoo.ifpack.adjoint.xml ${CMAKE_CURRENT_BINARY_DIR}/FourByFour.aztecoo.ifpack.adjoint.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/_FourByFour.aztecoo.ml.xml ${CMAKE_CURRENT_BINARY_DIR}/FourByFour.aztecoo.ml.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/_FourByFour.belos.np.xml ${CMAKE_CURRENT_BINARY_DIR}/FourByFour.belos.np.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/_FourByFour.belos.ifpack.xml ${CMAKE_CURRENT_BINARY_DIR}/FourByFour.belos.ifpack.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/_FourByFour.belos.ml.xml ${CMAKE_CURRENT_BINARY_DIR}/FourByFour.belos.ml.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/_BelosGCRODRTest.xml ${CMAKE_CURRENT_BINARY_DIR}/BelosGCRODRTest.xml COPYONLY)

#/#

add_test(test_single_stratimikos_solver_driver_amesos_1 ./test_single_stratimikos_solver_driver --input-file=FourByFour.xml --show-timer-summary)
add_test(test_single_stratimikos_solver_driver_amesos_2 ./test_single_stratimikos_solver_driver --input-file=FourByFour.amesos.xml --show-timer-summary)
add_test(test_single_stratimikos_solver_driver_aztecoo_np ./test_single_stratimikos_solver_driver --input-file=FourByFour.aztecoo.np.xml --show-timer-summary)
add_test(test_single_stratimikos_solver_driver_aztecoo_ifpack_1 ./test_single_stratimikos_solver_driver --input-file=FourByFour.aztecoo.ifpack.xml --show-timer-summary)
add_test(test_single_stratimikos_solver_driver_aztecoo_ifpack_2 ./test_single_stratimikos_solver_driver --input-file=FourByFour.aztecoo.ifpack.adjoint.xml --show-timer-summary)
add_test(test_single_stratimikos_solver_driver_aztecoo_ml ./test_single_stratimikos_solver_driver --input-file=FourByFour.aztecoo.ml.xml --show-timer-summary)
add_test(test_single_stratimikos_solver_driver_belos_np ./test_single_stratimikos_solver_driver --input-file=FourByFour.belos.np.xml --show-timer-summary)
add_test(test_single_stratimikos_solver_driver_belos_ifpack ./test_single_stratimikos_solver_driver --input-file=FourByFour.belos.ifpack.xml --show-timer-summary)
add_test(test_single_stratimikos_solver_driver_belos_ml ./test_single_stratimikos_solver_driver --input-file=FourByFour.belos.ml.xml --show-timer-summary)


####
INCLUDE(Dart)
INCLUDE(CPack)

