include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#Add Trilinos information to the include and link lines
include_directories(${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS} )
link_directories(${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS} )
# /Library/Frameworks/QtCore.framework /Library/Frameworks/QtGui.framework)

#set trilinos libraries to link (LINK_LIBRARIES)
set(LINK_LIBRARIES ${Stratimikos_LIBRARIES} ${Thyra_LIBRARIES} ${Epetraext_LIBRARIES} ${Epetra_LIBRARIES} ${Tpetra_LIBRARIES} ${Teuchos_LIBRARIES})


################################################################################
#
# MixedOrderPhysicsBasedPreconditioner
#


add_executable(Stratimikos_Preconditioner MixedOrderPhysicsBasedPreconditioner.cpp)
target_link_libraries(Stratimikos_Preconditioner ${LINK_LIBRARIES})

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

####
INCLUDE(Dart)
INCLUDE(CPack)
