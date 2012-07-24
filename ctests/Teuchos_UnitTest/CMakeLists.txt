include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#Add Trilinos information to the include and link lines
include_directories(${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS} )
link_directories(${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS} )
# /Library/Frameworks/QtCore.framework /Library/Frameworks/QtGui.framework)

#set trilinos libraries to link (LINK_LIBRARIES)
set(LINK_LIBRARIES ${Epetra_LIBRARIES} ${Galeri_LIBRARIES} ${Teuchos_LIBRARIES} ${Ifpack_LIBRARIES} ${Belos_LIBRARIES} ${Tpetra_LIBRARIES} ${Epetraext_LIBRARIES} ${TEUCHOS_STD_UNIT_TEST_MAIN})

##########################

set(PACKAGE_NAME Teuchos)

add_executable(Tec_UnitTest 
    Int_UnitTests.cpp
    TemplateFunc_UnitTests.cpp
    vector_UnitTests.cpp
    Teuchos_StandardUnitTestMain.cpp)
target_link_libraries(Tec_UnitTest  ${LINK_LIBRARIES})
add_test(Tec_UnitTest ${EXECUTABLE_OUTPUT_PATH}/Tec_UnitTest )
add_test(Tec_UnitTest_mpi mpiexec Tec_UnitTest )

#IF (${PACKAGE_NAME}_ENABLE_COMPLEX AND ${PACKAGE_NAME}_ENABLE_FLOAT)
#  SET(PASS_STRING "Summary: total = 18, run = 17, passed = 17, failed = 0")
#ELSEIF (${PACKAGE_NAME}_ENABLE_COMPLEX AND NOT ${PACKAGE_NAME}_ENABLE_FLOAT)
#  SET(PASS_STRING "Summary: total = 16, run = 15, passed = 15, failed = 0")
#ELSEIF (NOT ${PACKAGE_NAME}_ENABLE_COMPLEX AND ${PACKAGE_NAME}_ENABLE_FLOAT)
#  SET(PASS_STRING "Summary: total = 16, run = 15, passed = 15, failed = 0")
#ELSEIF (NOT ${PACKAGE_NAME}_ENABLE_COMPLEX AND NOT ${PACKAGE_NAME}_ENABLE_FLOAT)
#  SET(PASS_STRING "Summary: total = 15, run = 14, passed = 14, failed = 0")
#ENDIF()

add_test(Tec_UnitTest_minus_one 
  ${EXECUTABLE_OUTPUT_PATH}/Tec_UnitTest "--not-unit-test=vector_float_constAt_UnitTest")
#set_tests_properties(Tec_UnitTest_minus_one PROPERTIES PASS_REGULAR_EXPRESSION "${PASS_STRING}")

add_executable(Tec_BadUnitTest 
    Int_UnitTests.cpp
    TemplateFunc_UnitTests.cpp
    Failing_UnitTest.cpp
    Teuchos_StandardUnitTestMain.cpp)
target_link_libraries(Tec_BadUnitTest ${LINK_LIBRARIES})

add_test(Tec_BadUnitTest_Final   
  ${EXECUTABLE_OUTPUT_PATH}/Tec_BadUnitTest "--show-test-details=FINAL_RESULTS")
set_tests_properties(Tec_BadUnitTest_Final PROPERTIES PASS_REGULAR_EXPRESSION "Int_BadAssignment_UnitTest")

add_test(Tec_BadUnitTest_Result   
  ${EXECUTABLE_OUTPUT_PATH}/Tec_BadUnitTest "--teuchos-suppress-startup-banner")
set_tests_properties(Tec_BadUnitTest_Result PROPERTIES PASS_REGULAR_EXPRESSION "End Result: TEST FAILED")

add_test(Tec_BadUnitTest_Result_total   
  ${EXECUTABLE_OUTPUT_PATH}/Tec_BadUnitTest "--teuchos-suppress-startup-banner")
set_tests_properties(Tec_BadUnitTest_Result_total PROPERTIES PASS_REGULAR_EXPRESSION ", failed = 2")

add_test(Tec_BadUnitTest_BadAssignment    
  ${EXECUTABLE_OUTPUT_PATH}/Tec_BadUnitTest "--show-test-details=FINAL_RESULTS --group-name=Int --test-name=BadAssignment  --show-test-details=TEST_NAMES --group-name=Int --test-name=BadAssignment")
set_tests_properties(Tec_BadUnitTest_BadAssignment PROPERTIES PASS_REGULAR_EXPRESSION "[FAILED]")


INCLUDE(Dart)
INCLUDE(CPack)

