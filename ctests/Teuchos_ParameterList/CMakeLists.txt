include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#Add Trilinos information to the include and link lines
include_directories(${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS} )
link_directories(${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS} )

#create a helper library 
add_library(teuchos_xml_pl_test_helpers Teuchos_XMLParameterListTestHelpers.cpp Teuchos_XMLParameterListTestHelpers.hpp)


#set trilinos libraries to link (LINK_LIBRARIES)
set(LINK_LIBRARIES teuchos_xml_pl_test_helpers ${Teuchos_LIBRARIES})
#)



#////////////////////////////////////////////////////////////////////////////////////////////////////////
#this execution of configure file moves a file. the first parameter is the location of the target. the second is the desired destination. This can move a file from the source directory to the build directory so that the file can be accessed by the executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/BadRootElement.xml ${CMAKE_CURRENT_BINARY_DIR}/BadRootElement.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/BadParameterListElement.xml ${CMAKE_CURRENT_BINARY_DIR}/BadParameterListElement.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/NoNameAttribute.xml ${CMAKE_CURRENT_BINARY_DIR}/NoNameAttribute.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/NoTypeAttribute.xml ${CMAKE_CURRENT_BINARY_DIR}/NoTypeAttribute.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/NoValueAttribute.xml ${CMAKE_CURRENT_BINARY_DIR}/NoValueAttribute.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CantFindParameterEntryConverter.xml ${CMAKE_CURRENT_BINARY_DIR}/CantFindParameterEntryConverter.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/DuplicateParameterIDs.xml ${CMAKE_CURRENT_BINARY_DIR}/DuplicateParameterIDs.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/_input.xml ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)

# the command add executable is useg to generate executables using cmake. the parameters are first the desired name of the executable followed by any and all source codes to generate it.
add_executable(Tec_PL_maintest cxx_main.cpp)
# this command links the necessary libraries to the executable so that it functions correctly
target_link_libraries(Tec_PL_maintest  ${LINK_LIBRARIES})
# this command adds a test that can be executed by the ctest command. the first parameter is the name of the test. following is the command that the test runs (both executable and arguments)
add_test(Tec_PL_maintest ${EXECUTABLE_OUTPUT_PATH}/Tec_PL_maintest --verbose)


add_executable(Tec_PL_unitTest ParameterList_UnitTest.cpp Teuchos_StandardUnitTestMain.cpp)
target_link_libraries(Tec_PL_unitTest  ${LINK_LIBRARIES})
add_test(Tec_PL_unitTest ${EXECUTABLE_OUTPUT_PATH}/Tec_PL_unitTest )


add_executable(Tec_PEntry_unitTest ParameterEntry_UnitTest.cpp Teuchos_StandardUnitTestMain.cpp)
target_link_libraries(Tec_PEntry_unitTest  ${LINK_LIBRARIES})
add_test(Tec_PEntry_unitTest ${EXECUTABLE_OUTPUT_PATH}/Tec_PEntry_unitTest )


add_executable(Tec_PL_serialTest ParameterList_SerializationTest.cpp Teuchos_StandardUnitTestMain.cpp)
target_link_libraries(Tec_PL_serialTest  ${LINK_LIBRARIES})
add_test(Tec_PL_serialTest ${EXECUTABLE_OUTPUT_PATH}/Tec_PL_serialTest )


add_executable(Tec_PL_unitTest_par ParameterList_UnitTest_Parallel.cpp Teuchos_StandardUnitTestMain.cpp)
target_link_libraries(Tec_PL_unitTest_par  ${LINK_LIBRARIES})
add_test(Tec_PL_unitTest_par ${EXECUTABLE_OUTPUT_PATH}/Tec_PL_unitTest_par )

# this include enables and allows ctesting to be set up.
INCLUDE(Dart)
# this include allows the use of cpack to set up an installer
INCLUDE(CPack)
