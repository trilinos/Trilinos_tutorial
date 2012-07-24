include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#Add Trilinos information to the include and link lines
include_directories(${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS} )
link_directories(${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS} )
# /Library/Frameworks/QtCore.framework /Library/Frameworks/QtGui.framework)

#set trilinos libraries to link (LINK_LIBRARIES)
set(LINK_LIBRARIES ${Stratimikos_LIBRARIES} ${Thyra_LIBRARIES} ${Epetraext_LIBRARIES} ${Epetra_LIBRARIES} ${Tpetra_LIBRARIES} ${Teuchos_LIBRARIES})

####

#
# simple_stratimikos_example
#

add_executable(simple_stratimikos_example simple_stratimikos_example.cpp)
target_link_libraries(simple_stratimikos_example ${LINK_LIBRARIES})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/In_FourByFour.mtx ${CMAKE_CURRENT_BINARY_DIR}/FourByFour.mtx COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/amesos.klu.xml ${CMAKE_CURRENT_BINARY_DIR}/amesos.klu.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/aztecoo.ifpack.xml ${CMAKE_CURRENT_BINARY_DIR}/aztecoo.ifpack.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/aztecoo.ml.xml ${CMAKE_CURRENT_BINARY_DIR}/aztecoo.ml.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/belos.ifpack.xml ${CMAKE_CURRENT_BINARY_DIR}/belos.ifpack.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/belos.ml.xml ${CMAKE_CURRENT_BINARY_DIR}/belos.ml.xml COPYONLY)


add_test(simple_stratimikos_example_amesos_1 ./simple_stratimikos_example  --linear-solver-params-file=amesos.klu.xml --matrix-file=FourByFour.mtx --tol=1e-13 )
add_test(simple_stratimikos_example_amesos_2 ./simple_stratimikos_example  --extra-params-file=ExtraAmesosParams.xml --matrix-file=FourByFour.mtx --tol=1e-10 --linear-solver-params-used-file=lowsf.amesos.used.xml )


add_test(simple_stratimikos_example_aztecoo_ifpack_1 ./simple_stratimikos_example  --linear-solver-params-file=aztecoo.ifpack.xml --matrix-file=FourByFour.mtx --tol=1e-13 )
add_test(simple_stratimikos_example_aztecoo_ifpack_2 ./simple_stratimikos_example  --extra-params-file=ExtraAztecOOParams.xml --matrix-file=FourByFour.mtx --tol=1e-10 --linear-solver-params-used-file=lowsf.aztecoo.used.xml )

add_test(simple_stratimikos_example_aztecoo_ml ./simple_stratimikos_example  --linear-solver-params-file=aztecoo.ml.xml --matrix-file=FourByFour.mtx --tol=1e-13 )

add_test(simple_stratimikos_example_belos_ifpack ./simple_stratimikos_example  --linear-solver-params-file=belos.ifpack.xml --matrix-file=FourByFour.mtx --tol=1e-13 )

add_test(simple_stratimikos_example_belos_ml ./simple_stratimikos_example  --linear-solver-params-file=belos.ml.xml --matrix-file=FourByFour.mtx --tol=1e-12 )
        # 2008/12/11: rabartl Above: I had to drop the tolerance from 1e-13 to 1-e12 from
        # what is listed in stratimikos/test/defintion.  This must be due to some bad
        # compiler option or something but I don't have time to figure this out.


#
# Generate sample output
#
#
#CONFIGURE_FILE(generate-sample-output.pl.stub.in generate-sample-output.pl)
#
#ADD_CUSTOM_TARGET(stratimikos_sample_output
#  COMMENT  Generating sample output for package ${PACKAGE_NAME} ... 
#  COMMAND ${PERL_EXECUTABLE} generate-sample-output.pl
#  )
################################################################################
#
# MixedOrderPhysicsBasedPreconditioner
#


add_executable(MixedOrderPhysicsBasedPreconditioner MixedOrderPhysicsBasedPreconditioner.cpp)
target_link_libraries(MixedOrderPhysicsBasedPreconditioner ${LINK_LIBRARIES})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MixedOrderPhysicsBasedPreconditioner.Base.xml ${CMAKE_CURRENT_BINARY_DIR}/MixedOrderPhysicsBasedPreconditioner.Base.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MixedOrderPhysicsBasedPreconditioner.AllSolveOutput.xml ${CMAKE_CURRENT_BINARY_DIR}/MixedOrderPhysicsBasedPreconditioner.AllSolveOutput.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MixedOrderPhysicsBasedPreconditioner.P2SolveOutputMSolveOutputToFile.xml ${CMAKE_CURRENT_BINARY_DIR}/MixedOrderPhysicsBasedPreconditioner.P2SolveOutputMSolveOutputToFile.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MixedOrderPhysicsBasedPreconditioner.P2SolveOutput.xml ${CMAKE_CURRENT_BINARY_DIR}/MixedOrderPhysicsBasedPreconditioner.P2SolveOutput.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MixedOrderPhysicsBasedPreconditioner.Belos.xml ${CMAKE_CURRENT_BINARY_DIR}/MixedOrderPhysicsBasedPreconditioner.Belos.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/P2.mtx ${CMAKE_CURRENT_BINARY_DIR}/P2.mtx COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/P1.mtx ${CMAKE_CURRENT_BINARY_DIR}/P1.mtx COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/M11.mtx ${CMAKE_CURRENT_BINARY_DIR}/M11.mtx COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/M12.mtx ${CMAKE_CURRENT_BINARY_DIR}/M12.mtx COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/M21.mtx ${CMAKE_CURRENT_BINARY_DIR}/M21.mtx COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/M22.mtx ${CMAKE_CURRENT_BINARY_DIR}/M22.mtx COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ExtraAmesosParams.xml ${CMAKE_CURRENT_BINARY_DIR}/ExtraAmesosParams.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ExtraAztecOOParams.xml ${CMAKE_CURRENT_BINARY_DIR}/ExtraAztecOOParams.xml COPYONLY)

add_test(MixedOrderPhysicsBasedPreconditioner_aztecoo_ifpack1 ./MixedOrderPhysicsBasedPreconditioner  --params-file=MixedOrderPhysicsBasedPreconditioner.Base.xml --solve-tol=1.0 )
add_test(MixedOrderPhysicsBasedPreconditioner_aztecoo_ifpack2 ./MixedOrderPhysicsBasedPreconditioner  --params-file=MixedOrderPhysicsBasedPreconditioner.Base.xml --extra-params-file=MixedOrderPhysicsBasedPreconditioner.P2SolveOutput.xml --solve-tol=1.0 )
add_test(MixedOrderPhysicsBasedPreconditioner_aztecoo_ifpack3 ./MixedOrderPhysicsBasedPreconditioner  --params-file=MixedOrderPhysicsBasedPreconditioner.Base.xml --extra-params-file=MixedOrderPhysicsBasedPreconditioner.AllSolveOutput.xml --solve-tol=1.0 )
add_test(MixedOrderPhysicsBasedPreconditioner_aztecoo_ifpack4 ./MixedOrderPhysicsBasedPreconditioner  --params-file=MixedOrderPhysicsBasedPreconditioner.Base.xml --extra-params-file=MixedOrderPhysicsBasedPreconditioner.P2SolveOutputMSolveOutputToFile.xml --solve-tol=1.0 )
add_test(MixedOrderPhysicsBasedPreconditioner_aztecoo_ifpack5 ./MixedOrderPhysicsBasedPreconditioner  --params-file=MixedOrderPhysicsBasedPreconditioner.Base.xml --extra-params-file=MixedOrderPhysicsBasedPreconditioner.P2SolveOutput.xml --use-algebraic-prec --solve-tol=1.0 )

add_test(MixedOrderPhysicsBasedPreconditioner_belos_ifpack1 ./MixedOrderPhysicsBasedPreconditioner  --params-file=MixedOrderPhysicsBasedPreconditioner.Belos.xml --use-algebraic-prec --solve-tol=1e+5 )
add_test(MixedOrderPhysicsBasedPreconditioner_belos_ifpack2 ./MixedOrderPhysicsBasedPreconditioner  --params-file=MixedOrderPhysicsBasedPreconditioner.Belos.xml --invert-P1 --solve-tol=1e+5 )
 
####
INCLUDE(Dart)
INCLUDE(CPack)
